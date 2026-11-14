package utils

import (
	"encoding/json"
	"testing"
)

func TestRepairJSONObjectMissingInnerBrace(t *testing.T) {
	// Real Gemini reply shape: the arguments object's closing brace is missing.
	in := `{"tool_calls":[{"name":"bash","arguments":{"command":"echo \"〜てもいいですか\" > a.md"}]}`
	out := RepairJSONObject(in)
	if out == "" {
		t.Fatal("expected repaired JSON")
	}
	var v struct {
		ToolCalls []struct {
			Name      string            `json:"name"`
			Arguments map[string]string `json:"arguments"`
		} `json:"tool_calls"`
	}
	if err := json.Unmarshal([]byte(out), &v); err != nil {
		t.Fatalf("repaired JSON does not decode: %v\n%s", err, out)
	}
	if len(v.ToolCalls) != 1 || v.ToolCalls[0].Name != "bash" || v.ToolCalls[0].Arguments["command"] == "" {
		t.Fatalf("unexpected decode: %+v", v)
	}
}

func TestRepairJSONObjectTruncatedAndTrailingText(t *testing.T) {
	cases := map[string]string{
		"missing end closers": `{"tool_calls":[{"name":"ls","arguments":{"path":"."`,
		"unterminated string": `{"content":"hello wor`,
		"trailing prose":      `Sure! {"content":"done"} hope that helps }`,
		"raw newline":         "{\"content\":\"line1\nline2\"}",
	}
	for name, in := range cases {
		if out := RepairJSONObject(in); out == "" || !json.Valid([]byte(out)) {
			t.Errorf("%s: got %q", name, out)
		}
	}
	if out := RepairJSONObject("no json here"); out != "" {
		t.Errorf("expected empty for non-JSON, got %q", out)
	}
}

func TestParseTextToolCalls(t *testing.T) {
	in := `[Call Tool: bash with ID call_1794684755123456789_0 and Input: {"command":"ls -la jlpt_guide/"}]`
	calls := ParseTextToolCalls(in)
	if len(calls) != 1 || calls[0].Name != "bash" || calls[0].Arguments != `{"command":"ls -la jlpt_guide/"}` {
		t.Fatalf("unexpected calls: %+v", calls)
	}

	gem := `[Call Tool: read with Input: {"path":"a.txt"}]` + "\n" + `[Call Tool: write with ID x and Input: {"path":"b.txt","content":"hi"}]`
	if calls := ParseTextToolCalls(gem); len(calls) != 2 || calls[0].Name != "read" || calls[1].Name != "write" {
		t.Fatalf("unexpected calls: %+v", calls)
	}

	if calls := ParseTextToolCalls("just a normal answer"); calls != nil {
		t.Fatalf("expected no calls, got %+v", calls)
	}
}

func TestLooksLikeToolCallAttempt(t *testing.T) {
	yes := []string{
		`{"tool_calls":[{"name":"bash"`,
		"```json\n{\"tool_calls\": [\n```",
		`[Call Tool: bash with ID x and Input: {broken`,
	}
	for _, s := range yes {
		if !LooksLikeToolCallAttempt(s) {
			t.Errorf("expected attempt: %q", s)
		}
	}
	if LooksLikeToolCallAttempt("The file was created. Anything else?") {
		t.Error("plain text flagged as tool attempt")
	}
}

func TestUnwrapAutoLinks(t *testing.T) {
	cases := map[string]string{
		// real reply from the pi session that broke bash with "syntax error near '('"
		`curl -sL [https://api.github.com/repos/vercel/geist-font/releases/latest](https://api.github.com/repos/vercel/geist-font/releases/latest)`: `curl -sL https://api.github.com/repos/vercel/geist-font/releases/latest`,
		`see [github.com/a/b](https://github.com/a/b) now`: `see github.com/a/b now`,
		`keep [the docs](https://example.com/docs) link`:  `keep [the docs](https://example.com/docs) link`,
		`no links here`: `no links here`,
	}
	for in, want := range cases {
		if got := UnwrapAutoLinks(in); got != want {
			t.Errorf("UnwrapAutoLinks(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestCleanToolBridgeReplyOnlyTouchesToolCalls(t *testing.T) {
	call := `{"tool_calls":[{"name":"bash","arguments":{"command":"curl [https://x.dev/a](https://x.dev/a)"}}]}`
	if got := CleanToolBridgeReply(call); got != `{"tool_calls":[{"name":"bash","arguments":{"command":"curl https://x.dev/a"}}]}` {
		t.Errorf("got %q", got)
	}
	prose := `Read [https://x.dev/a](https://x.dev/a) for more.`
	if got := CleanToolBridgeReply(prose); got != prose {
		t.Errorf("prose changed: %q", got)
	}
}

func TestLooksLikeDeferral(t *testing.T) {
	for _, s := range []string{
		"I don't have any tool calls remaining in this turn, so send a quick reply and I'll do it.",
		"I’ve hit my tool-call limit for now.",
		`I cloned the repo. Send a quick reply (e.g., "start it") and I'll spin it up.`,
		`Just reply "go" and I will create the remaining 5 files.`,
	} {
		if !LooksLikeDeferral(s) {
			t.Errorf("not detected: %q", s)
		}
	}
	for _, s := range []string{
		"All 10 files are written and cargo test passes. Want me to add benchmarks?",
		"Done. Let me know if you need anything else.",
		"Floci is running at http://localhost:4566.",
	} {
		if LooksLikeDeferral(s) {
			t.Errorf("false positive: %q", s)
		}
	}
}
