package providers

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

// A server that sends part of a reply and then goes silent must be cut off
// after the stall timeout, while one that keeps sending slowly must not be.
func TestStallResetReader(t *testing.T) {
	const stall = 300 * time.Millisecond

	run := func(t *testing.T, handler http.HandlerFunc) (int, bool, error) {
		t.Helper()
		srv := httptest.NewServer(handler)
		defer srv.Close()

		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		var stalled atomic.Bool
		timer := time.AfterFunc(stall, func() { stalled.Store(true); cancel() })
		defer timer.Stop()

		req, _ := http.NewRequestWithContext(ctx, "GET", srv.URL, nil)
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			return 0, stalled.Load(), err
		}
		defer resp.Body.Close()
		b, err := io.ReadAll(&stallResetReader{r: resp.Body, timer: timer, d: stall})
		return len(b), stalled.Load(), err
	}

	t.Run("silent server is cut off", func(t *testing.T) {
		done := make(chan struct{})
		defer close(done)
		start := time.Now()
		n, stalled, err := run(t, func(w http.ResponseWriter, r *http.Request) {
			_, _ = w.Write([]byte("partial"))
			w.(http.Flusher).Flush()
			select { // hang like a stalled Gemini reply
			case <-done:
			case <-r.Context().Done():
			}
		})
		if err == nil || !stalled {
			t.Fatalf("expected a stall, got n=%d stalled=%v err=%v", n, stalled, err)
		}
		if took := time.Since(start); took > 3*time.Second {
			t.Errorf("stall detection took %s", took)
		}
	})

	t.Run("slow but steady server is not cut off", func(t *testing.T) {
		n, stalled, err := run(t, func(w http.ResponseWriter, r *http.Request) {
			for i := 0; i < 5; i++ { // total 1s, far beyond the 300ms stall window
				_, _ = w.Write([]byte("chunk"))
				w.(http.Flusher).Flush()
				time.Sleep(200 * time.Millisecond)
			}
		})
		if err != nil || stalled || n != 25 {
			t.Fatalf("got n=%d stalled=%v err=%v", n, stalled, err)
		}
	})
}
