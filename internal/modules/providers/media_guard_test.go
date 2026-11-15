package providers

import (
	"errors"
	"strings"
	"testing"
)

func TestStartedMediaGeneration(t *testing.T) {
	cases := []struct {
		name string
		resp *Response
		want bool
	}{
		{"nil", nil, false},
		{"video card", &Response{Videos: []Video{{URL: "https://x"}}}, true},
		{"audio card", &Response{Audios: []Audio{{URL: "https://x"}}}, true},
		{"video ack", &Response{Text: "I'm generating your video. This can take a few minutes, so check back soon."}, true},
		{"video ack 2", &Response{Text: "Generating your video now..."}, true},
		{"music ack", &Response{Text: "I'll create a track for you with an upbeat feel."}, true},
		{"ready soon", &Response{Text: "Sure! Your video will be ready in a few minutes."}, true},
		{"quota notice", &Response{Text: "I can create more videos as soon as your limit resets. Check your usage in Settings."}, true},
		{"limit discussion", &Response{Text: "YouTube has an upload limit for new accounts."}, false},
		{"plain answer", &Response{Text: "The capital of France is Paris."}, false},
		{"mentions video", &Response{Text: "You can watch the music video on YouTube."}, false},
		{"how-to", &Response{Text: "To make a video in DaVinci Resolve, import your clips first."}, false},
		{"tool json", &Response{Text: `{"tool_calls":[{"name":"bash","arguments":{"command":"ffmpeg -i in.mp4 out.webm"}}]}`}, false},
		{"long content", &Response{Text: "I'm generating your video summary below.\n" + strings.Repeat("Details. ", 100)}, false},
	}
	for _, tc := range cases {
		if got := StartedMediaGeneration(tc.resp); got != tc.want {
			t.Errorf("%s: got %v, want %v", tc.name, got, tc.want)
		}
	}
}

func TestIsChatNotContinuableError(t *testing.T) {
	if !isChatNotContinuableError(errors.New("Google Gemini Web returned an error (BardErrorInfo code [13 1097]). This usually...")) {
		t.Fatal("1097 not detected")
	}
	if isChatNotContinuableError(errors.New("BardErrorInfo code [13 1185]")) || isChatNotContinuableError(nil) {
		t.Fatal("false positive")
	}
}

func TestWithTextOnly(t *testing.T) {
	cfg := &GenerateConfig{}
	WithTextOnly()(cfg)
	if !cfg.TextOnly {
		t.Fatal("TextOnly not set")
	}
}
