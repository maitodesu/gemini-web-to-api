package providers

import (
	"errors"
	"io"
	"time"
)

// generateStallTimeout is how long a generate attempt may go without
// receiving any data from Gemini Web before it is cancelled and retried.
// Gemini streams its reply in chunks, so long answers keep resetting the
// timer; only a connection that has gone silent is cut. Without this, one
// stalled attempt used up the whole request deadline and the retry never ran.
var generateStallTimeout = 120 * time.Second

var errGenerateStalled = errors.New("generate response stalled")

// stallResetReader pushes the stall timer back every time data arrives.
type stallResetReader struct {
	r     io.Reader
	timer *time.Timer
	d     time.Duration
}

func (s *stallResetReader) Read(p []byte) (int, error) {
	n, err := s.r.Read(p)
	if n > 0 {
		s.timer.Reset(s.d)
	}
	return n, err
}
