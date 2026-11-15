package openai

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"gemini-web-to-api/internal/modules/openai/dto"
	"gemini-web-to-api/internal/modules/providers"
)

// TestSessionReuseSendsOnlyNewMessages checks that a follow-up request goes
// into the same Gemini chat with only its new messages.
func TestSessionReuseSendsOnlyNewMessages(t *testing.T) {
	var prompts []string
	var convs []*providers.SessionMetadata
	turn := 0
	fake := &fakeGemini{}
	service := newToolBridgeTestService(t, fake)
	service.client.SetSessionReuse(true)
	service.generate = func(_ context.Context, prompt string, opts ...providers.GenerateOption) (*providers.Response, error) {
		cfg := &providers.GenerateConfig{}
		for _, o := range opts {
			o(cfg)
		}
		prompts = append(prompts, prompt)
		convs = append(convs, cfg.Conversation)
		turn++
		return &providers.Response{Text: "reply", Model: "gemini-3.6-flash",
			Metadata: map[string]any{"cid": "c_1", "rid": "r_" + string(rune('0'+turn)), "rcid": "rc"}}, nil
	}

	request := func(body string) {
		t.Helper()
		var req dto.ChatCompletionRequest
		if err := json.Unmarshal([]byte(body), &req); err != nil {
			t.Fatal(err)
		}
		if _, err := service.CreateChatCompletion(context.Background(), req); err != nil {
			t.Fatal(err)
		}
	}
	request(`{"model":"gemini-3.6-flash","messages":[{"role":"system","content":"SYS"},{"role":"user","content":"FIRST"}]}`)
	request(`{"model":"gemini-3.6-flash","messages":[{"role":"system","content":"SYS"},{"role":"user","content":"FIRST"},
		{"role":"assistant","content":"reply"},{"role":"user","content":"SECOND"}]}`)

	if convs[0] != nil || !strings.Contains(prompts[0], "FIRST") {
		t.Fatalf("first turn: conv=%v prompt=%q", convs[0], prompts[0])
	}
	if convs[1] == nil || convs[1].ConversationID != "c_1" || convs[1].ResponseID != "r_1" {
		t.Fatalf("second turn should continue c_1/r_1, got %+v", convs[1])
	}
	if strings.Contains(prompts[1], "FIRST") || strings.Contains(prompts[1], "SYS") || !strings.Contains(prompts[1], "SECOND") {
		t.Fatalf("second prompt should hold only the new message, got %q", prompts[1])
	}
}
