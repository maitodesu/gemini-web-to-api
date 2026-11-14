package openai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"gemini-web-to-api/internal/modules/openai/dto"
	"gemini-web-to-api/internal/modules/providers"

	"github.com/gofiber/fiber/v3"
	"go.uber.org/zap"
)

// doneToolRequestJSON is the forced tool call from issue #96: a required string
// and a required nested object with required fields.
const doneToolRequestJSON = `{
	"model": "gemini-3.6-flash",
	"messages": [{"role":"user","content":"Call the done tool."}],
	"tools": [{"type":"function","function":{"name":"done","description":"Finish with answer and document",
		"parameters":{"type":"object",
			"properties":{"answer":{"type":"string"},
				"document":{"type":"object","properties":{"title":{"type":"string"},"body":{"type":"string"}},"required":["title","body"]}},
			"required":["answer","document"]}}}],
	"tool_choice": %s
}`

const forcedDone = `{"type":"function","function":{"name":"done"}}`

const wantDoneArguments = `{"answer":"Done.","document":{"title":"Test","body":"# Hello\n\nSome markdown body."}}`

func doneToolRequest(t *testing.T, toolChoice string) dto.ChatCompletionRequest {
	t.Helper()
	var req dto.ChatCompletionRequest
	if err := json.Unmarshal([]byte(fmt.Sprintf(doneToolRequestJSON, toolChoice)), &req); err != nil {
		t.Fatal(err)
	}
	return req
}

// fakeGemini replays canned replies, repeating the last one, and records prompts.
type fakeGemini struct {
	replies []string
	prompts []string
}

func (f *fakeGemini) generate(_ context.Context, prompt string, _ ...providers.GenerateOption) (*providers.Response, error) {
	f.prompts = append(f.prompts, prompt)
	reply := f.replies[min(len(f.prompts), len(f.replies))-1]
	return &providers.Response{Text: reply, Model: "gemini-3.6-flash"}, nil
}

func newToolBridgeTestService(t *testing.T, fake *fakeGemini) *OpenAIService {
	t.Helper()
	client := providers.NewVideoTestClient([]providers.ModelInfo{{ID: "gemini-3.6-flash"}}, nil)
	t.Cleanup(func() { _ = client.Close() })
	service := NewOpenAIService(client, zap.NewNop())
	service.generate = fake.generate
	return service
}

func TestToolBridgeParsesLenientReplies(t *testing.T) {
	service := &OpenAIService{log: zap.NewNop()}
	req := doneToolRequest(t, forcedDone)

	tests := []struct {
		name  string
		reply string
	}{
		{
			name:  "raw newlines inside argument strings",
			reply: "{\"tool_calls\":[{\"name\":\"done\",\"arguments\":{\"answer\":\"Done.\",\"document\":{\"title\":\"Test\",\"body\":\"# Hello\n\nSome markdown body.\"}}}]}",
		},
		{
			name:  "OpenAI-style nested function with string arguments",
			reply: `{"tool_calls":[{"type":"function","function":{"name":"done","arguments":"{\"answer\":\"Done.\",\"document\":{\"title\":\"Test\",\"body\":\"# Hello\\n\\nSome markdown body.\"}}"}}]}`,
		},
		{
			name:  "args instead of arguments",
			reply: `{"tool_calls":[{"name":"done","args":{"answer":"Done.","document":{"title":"Test","body":"# Hello\n\nSome markdown body."}}}]}`,
		},
		{
			name:  "prose with braces before the JSON",
			reply: "Calling done with {answer, document}:\n" + `{"tool_calls":[{"name":"done","arguments":{"answer":"Done.","document":{"title":"Test","body":"# Hello\n\nSome markdown body."}}}]}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			calls, _, err := service.parseToolBridgeOutput(req, tt.reply)
			if err != nil {
				t.Fatal(err)
			}
			if len(calls) != 1 {
				t.Fatalf("got %d tool calls, want 1", len(calls))
			}
			if calls[0].Function.Name != "done" || calls[0].Function.Arguments != wantDoneArguments {
				t.Errorf("got %s(%s), want done(%s)", calls[0].Function.Name, calls[0].Function.Arguments, wantDoneArguments)
			}
		})
	}
}

func TestToolBridgeRejectsUnusableCalls(t *testing.T) {
	service := &OpenAIService{log: zap.NewNop()}
	req := doneToolRequest(t, forcedDone)

	tests := []struct {
		name    string
		reply   string
		wantErr string
	}{
		{
			name:    "empty arguments",
			reply:   `{"tool_calls":[{"name":"done","arguments":{}}]}`,
			wantErr: `missing required argument "answer"`,
		},
		{
			name:    "missing nested argument",
			reply:   `{"tool_calls":[{"name":"done","arguments":{"answer":"Done.","document":{"title":"Test"}}}]}`,
			wantErr: `missing required argument "document.body"`,
		},
		{
			name:    "arguments string that is not JSON",
			reply:   `{"tool_calls":[{"name":"done","arguments":"{answer: Done."}]}`,
			wantErr: "not valid JSON",
		},
		{
			name:    "arguments that are not an object",
			reply:   `{"tool_calls":[{"name":"done","arguments":["Done."]}]}`,
			wantErr: "must be a JSON object",
		},
		{
			name:    "only an unknown tool",
			reply:   `{"tool_calls":[{"name":"finish","arguments":{"answer":"Done."}}]}`,
			wantErr: "no tool call named",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			calls, _, err := service.parseToolBridgeOutput(req, tt.reply)
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("got calls=%v err=%v, want error containing %q", calls, err, tt.wantErr)
			}
		})
	}
}

func TestToolBridgeAllowsEmptyArgumentsWithoutRequiredParameters(t *testing.T) {
	service := &OpenAIService{log: zap.NewNop()}
	req := dto.ChatCompletionRequest{Tools: []dto.ToolDefinition{{
		Type:     "function",
		Function: dto.ToolFunctionDefinition{Name: "ping", Parameters: json.RawMessage(`{"type":"object","properties":{}}`)},
	}}}

	calls, _, err := service.parseToolBridgeOutput(req, `{"tool_calls":[{"name":"ping"}]}`)
	if err != nil {
		t.Fatal(err)
	}
	if len(calls) != 1 || calls[0].Function.Arguments != "{}" {
		t.Fatalf("got %v, want one ping call with {}", calls)
	}
}

func TestCreateChatCompletionRetriesUnusableToolCall(t *testing.T) {
	fake := &fakeGemini{replies: []string{
		"Sure, here is a one-sentence summary.",
		`{"tool_calls":[{"name":"done","arguments":{}}]}`,
		`{"tool_calls":[{"name":"done","arguments":` + wantDoneArguments + `}]}`,
	}}
	service := newToolBridgeTestService(t, fake)

	resp, err := service.CreateChatCompletion(context.Background(), doneToolRequest(t, forcedDone))
	if err != nil {
		t.Fatal(err)
	}
	if len(fake.prompts) != 3 {
		t.Fatalf("Gemini was called %d times, want 3", len(fake.prompts))
	}
	for _, prompt := range fake.prompts[1:] {
		if !strings.Contains(prompt, "Your previous reply could not be used") {
			t.Errorf("retry prompt does not explain the failure:\n%s", prompt)
		}
	}
	choice := resp.Choices[0]
	if choice.FinishReason != "tool_calls" || len(choice.Message.ToolCalls) != 1 {
		t.Fatalf("got finish_reason=%q tool_calls=%v", choice.FinishReason, choice.Message.ToolCalls)
	}
	if got := choice.Message.ToolCalls[0].Function.Arguments; got != wantDoneArguments {
		t.Errorf("arguments = %s, want %s", got, wantDoneArguments)
	}
}

func TestCreateChatCompletionFailsInsteadOfFabricatingToolCall(t *testing.T) {
	for _, toolChoice := range []string{forcedDone, `"required"`} {
		t.Run(toolChoice, func(t *testing.T) {
			fake := &fakeGemini{replies: []string{"I cannot call tools, but the answer is 42."}}
			service := newToolBridgeTestService(t, fake)

			resp, err := service.CreateChatCompletion(context.Background(), doneToolRequest(t, toolChoice))
			if !errors.Is(err, ErrInvalidToolCall) {
				t.Fatalf("got resp=%v err=%v, want ErrInvalidToolCall", resp, err)
			}
			if len(fake.prompts) != maxToolBridgeAttempts {
				t.Errorf("Gemini was called %d times, want %d", len(fake.prompts), maxToolBridgeAttempts)
			}
		})
	}
}

func TestCreateChatCompletionAutoToolChoiceKeepsPlainText(t *testing.T) {
	fake := &fakeGemini{replies: []string{"The answer is 42."}}
	service := newToolBridgeTestService(t, fake)

	resp, err := service.CreateChatCompletion(context.Background(), doneToolRequest(t, `"auto"`))
	if err != nil {
		t.Fatal(err)
	}
	if len(fake.prompts) != 1 {
		t.Errorf("Gemini was called %d times, want 1", len(fake.prompts))
	}
	choice := resp.Choices[0]
	if choice.FinishReason != "stop" || choice.Message.Content != "The answer is 42." || len(choice.Message.ToolCalls) != 0 {
		t.Fatalf("got finish_reason=%q content=%v tool_calls=%v", choice.FinishReason, choice.Message.Content, choice.Message.ToolCalls)
	}
}

func TestChatCompletionsReturnBadGatewayForInvalidToolCall(t *testing.T) {
	fake := &fakeGemini{replies: []string{`{"tool_calls":[{"name":"done","arguments":{}}]}`}}
	app := fiber.New()
	NewOpenAIController(newToolBridgeTestService(t, fake), zap.NewNop()).Register(app.Group("/openai/v1"))

	req := httptest.NewRequest(http.MethodPost, "/openai/v1/chat/completions",
		strings.NewReader(fmt.Sprintf(doneToolRequestJSON, forcedDone)))
	req.Header.Set("Content-Type", "application/json")
	resp, err := app.Test(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusBadGateway {
		t.Fatalf("status = %d, want 502; body = %s", resp.StatusCode, body)
	}
	if !strings.Contains(string(body), `missing required argument \"answer\"`) {
		t.Errorf("error body does not say what was wrong: %s", body)
	}
}

// Real failures seen from Gemini with the pi coding agent (tool_choice auto):
// a tool call missing one closing brace, and a call copied in the transcript
// "[Call Tool: ...]" form. Both used to reach the client as plain text, which
// ended the agent's turn.
func TestToolBridgeRecoversMalformedAgentReplies(t *testing.T) {
	service := &OpenAIService{log: zap.NewNop()}
	req := doneToolRequest(t, `"auto"`)

	tests := []struct {
		name  string
		reply string
	}{
		{
			name:  "missing closing brace of arguments",
			reply: `{"tool_calls":[{"name":"done","arguments":{"answer":"Done.","document":{"title":"Test","body":"# Hello\n\nSome markdown body."}}]}`,
		},
		{
			name:  "transcript form with ID",
			reply: `[Call Tool: done with ID call_1794684755123456789_0 and Input: ` + wantDoneArguments + `]`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			calls, _, err := service.parseToolBridgeOutput(req, tt.reply)
			if err != nil {
				t.Fatal(err)
			}
			if len(calls) != 1 || calls[0].Function.Name != "done" || calls[0].Function.Arguments != wantDoneArguments {
				t.Fatalf("got %+v", calls)
			}
		})
	}
}

func TestCreateChatCompletionAutoRetriesBrokenToolCallInsteadOfStopping(t *testing.T) {
	fake := &fakeGemini{replies: []string{
		`{"tool_calls":[{"name":"done","arguments":{"answer":"Done.", "document": [unrecoverable`,
		`{"tool_calls":[{"name":"done","arguments":` + wantDoneArguments + `}]}`,
	}}
	service := newToolBridgeTestService(t, fake)

	resp, err := service.CreateChatCompletion(context.Background(), doneToolRequest(t, `"auto"`))
	if err != nil {
		t.Fatal(err)
	}
	if len(fake.prompts) != 2 {
		t.Fatalf("Gemini was called %d times, want 2", len(fake.prompts))
	}
	choice := resp.Choices[0]
	if choice.FinishReason != "tool_calls" || len(choice.Message.ToolCalls) != 1 {
		t.Fatalf("got finish_reason=%q content=%v tool_calls=%v", choice.FinishReason, choice.Message.Content, choice.Message.ToolCalls)
	}
}

func TestToolBridgePromptSaysThereIsNoToolLimit(t *testing.T) {
	service := &OpenAIService{log: zap.NewNop()}
	prompt := service.buildToolBridgePrompt(doneToolRequest(t, `"auto"`), "user: hi")
	for _, want := range []string{"NO limit on how many tool calls", "not a count against any budget", "never as Markdown links"} {
		if !strings.Contains(prompt, want) {
			t.Errorf("bridge prompt is missing %q", want)
		}
	}
}

func TestCreateChatCompletionRetriesDeferral(t *testing.T) {
	deferral := `{"content":"I don't have any tool calls remaining in this turn, so send a quick reply (e.g., \"create them\") and I'll write them right away!"}`
	t.Run("then calls a tool", func(t *testing.T) {
		fake := &fakeGemini{replies: []string{deferral, `{"tool_calls":[{"name":"done","arguments":` + wantDoneArguments + `}]}`}}
		resp, err := newToolBridgeTestService(t, fake).CreateChatCompletion(context.Background(), doneToolRequest(t, `"auto"`))
		if err != nil {
			t.Fatal(err)
		}
		if len(fake.prompts) != 2 || !strings.Contains(fake.prompts[1], "handed the task back") {
			t.Fatalf("Gemini was called %d times, want 2 with the deferral note", len(fake.prompts))
		}
		if choice := resp.Choices[0]; choice.FinishReason != "tool_calls" {
			t.Fatalf("finish_reason = %q, want tool_calls", choice.FinishReason)
		}
	})
	t.Run("keeps deferring", func(t *testing.T) {
		fake := &fakeGemini{replies: []string{deferral}}
		resp, err := newToolBridgeTestService(t, fake).CreateChatCompletion(context.Background(), doneToolRequest(t, `"auto"`))
		if err != nil {
			t.Fatalf("a persistent deferral must come back as text, got %v", err)
		}
		if len(fake.prompts) != maxToolBridgeAttempts {
			t.Errorf("Gemini was called %d times, want %d", len(fake.prompts), maxToolBridgeAttempts)
		}
		if resp.Choices[0].FinishReason != "stop" {
			t.Errorf("finish_reason = %q, want stop", resp.Choices[0].FinishReason)
		}
	})
}
