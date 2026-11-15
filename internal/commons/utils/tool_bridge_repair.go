package utils

import (
	"encoding/json"
	"regexp"
	"strconv"
	"strings"
	"unicode/utf16"
	"unicode/utf8"
)

// RepairJSONObject tries to turn a slightly malformed JSON object written by a
// model into valid JSON. It starts at the first '{' and fixes the slips Gemini
// makes most often in long tool-call replies: a missing closing brace or
// bracket before another closer (`..."}]}` instead of `..."}}]}`), missing
// closers at the end, an unterminated final string, raw control characters
// inside strings, and trailing prose after the object. It returns "" when the
// result still is not valid JSON, so callers can treat it as a best effort.
func RepairJSONObject(text string) string {
	start := strings.Index(text, "{")
	if start < 0 {
		return ""
	}

	var b strings.Builder
	var stack []byte // expected closers
	inString := false
	escaped := false

scan:
	for i := start; i < len(text); i++ {
		ch := text[i]
		if inString {
			b.WriteByte(ch)
			switch {
			case escaped:
				escaped = false
			case ch == '\\':
				escaped = true
			case ch == '"':
				inString = false
			}
			continue
		}

		switch ch {
		case '"':
			inString = true
			b.WriteByte(ch)
		case '{':
			stack = append(stack, '}')
			b.WriteByte(ch)
		case '[':
			stack = append(stack, ']')
			b.WriteByte(ch)
		case '}', ']':
			if len(stack) == 0 {
				break scan // trailing text after the object
			}
			if stack[len(stack)-1] != ch {
				// Mismatched closer: if the expected one is further down the
				// stack, the model skipped the closers above it, so add them.
				idx := -1
				for j := len(stack) - 1; j >= 0; j-- {
					if stack[j] == ch {
						idx = j
						break
					}
				}
				if idx < 0 {
					continue // stray closer with nothing to close, drop it
				}
				for len(stack)-1 > idx {
					b.WriteByte(stack[len(stack)-1])
					stack = stack[:len(stack)-1]
				}
			}
			stack = stack[:len(stack)-1]
			b.WriteByte(ch)
			if len(stack) == 0 {
				break scan
			}
		default:
			b.WriteByte(ch)
		}
	}

	if inString {
		if escaped {
			// a dangling backslash would escape the closing quote
			s := b.String()
			b.Reset()
			b.WriteString(s[:len(s)-1])
		}
		b.WriteByte('"')
	}
	for len(stack) > 0 {
		b.WriteByte(stack[len(stack)-1])
		stack = stack[:len(stack)-1]
	}

	out := b.String()
	if json.Valid([]byte(out)) {
		return out
	}
	if escapedOut := EscapeJSONStringControlChars(out); json.Valid([]byte(escapedOut)) {
		return escapedOut
	}
	return ""
}

// TextToolCall is a tool call the model wrote as plain text in the transcript
// format the bridge uses for earlier calls, e.g.
// `[Call Tool: bash with ID call_1 and Input: {"command":"ls"}]`.
type TextToolCall struct {
	Name      string
	Arguments string // JSON object
}

var textToolCallRe = regexp.MustCompile(`\[Call Tool:\s*([A-Za-z0-9_.:\-]+)\s+(?:with ID\s+\S+\s+and\s+)?(?:with\s+)?Input:\s*`)

// ParseTextToolCalls extracts tool calls written in the `[Call Tool: ...]`
// transcript form. Gemini copies this form from the conversation history
// instead of answering with bridge JSON. Calls whose input is not a JSON
// object (even after repair) are skipped.
func ParseTextToolCalls(text string) []TextToolCall {
	matches := textToolCallRe.FindAllStringSubmatchIndex(text, -1)
	if len(matches) == 0 {
		return nil
	}
	calls := make([]TextToolCall, 0, len(matches))
	for _, m := range matches {
		name := text[m[2]:m[3]]
		rest := text[m[1]:]
		if !strings.HasPrefix(strings.TrimSpace(rest), "{") {
			continue
		}
		args := ExtractFirstJSONObject(rest)
		if args == "" || !json.Valid([]byte(args)) {
			args = RepairJSONObject(rest)
		}
		if args == "" {
			continue
		}
		calls = append(calls, TextToolCall{Name: name, Arguments: args})
	}
	return calls
}

// LooksLikeToolCallAttempt reports whether a reply that could not be parsed
// was still meant to be a tool call, so the bridge should ask the model again
// instead of handing the raw text to the client as a final answer.
func LooksLikeToolCallAttempt(text string) bool {
	t := strings.TrimSpace(StripCodeFence(text))
	return strings.Contains(t, `"tool_calls"`) || strings.Contains(t, "[Call Tool:")
}

var (
	looseContentStartRe = regexp.MustCompile(`^\s*\{\s*(?:"status"\s*:\s*"[A-Za-z_]*"\s*,\s*)?"content"\s*:\s*"`)
	looseContentEndRe   = regexp.MustCompile(`"\s*(?:,\s*"status"\s*:\s*"[A-Za-z_]*"\s*)?\}\s*$`)
)

// LooseContentReply recovers the text of a {"content":"..."} bridge reply that
// is not valid JSON. Gemini often leaves quotes inside the text unescaped
// (`Continuous Actions ("-ing")`), so a strict decode fails and the client
// would see the raw JSON with literal \n sequences. The string is taken
// from the opening quote to the closing quote before the final brace, and
// JSON escapes in it are decoded leniently. Replies with tool calls are left
// alone: their arguments need a real decode.
func LooseContentReply(text string) (string, bool) {
	t := strings.TrimSpace(StripCodeFence(text))
	if strings.Contains(t, `"tool_calls"`) {
		return "", false
	}
	start := looseContentStartRe.FindStringIndex(t)
	end := looseContentEndRe.FindStringIndex(t)
	if start == nil || end == nil || end[0] < start[1] {
		return "", false
	}
	return strings.TrimSpace(unescapeJSONLenient(t[start[1]:end[0]])), true
}

// unescapeJSONLenient decodes JSON string escapes and keeps anything else,
// including bare quotes and unknown escapes, as written.
func unescapeJSONLenient(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c != '\\' || i+1 >= len(s) {
			b.WriteByte(c)
			continue
		}
		i++
		switch s[i] {
		case 'n':
			b.WriteByte('\n')
		case 't':
			b.WriteByte('\t')
		case 'r':
			b.WriteByte('\r')
		case 'b':
			b.WriteByte('\b')
		case 'f':
			b.WriteByte('\f')
		case '"', '\\', '/':
			b.WriteByte(s[i])
		case 'u':
			if i+4 < len(s) {
				if r, err := strconv.ParseUint(s[i+1:i+5], 16, 32); err == nil {
					ru := rune(r)
					// Join a surrogate pair (😀) into one rune.
					if utf16.IsSurrogate(ru) && i+10 < len(s) && s[i+5] == '\\' && s[i+6] == 'u' {
						if r2, err := strconv.ParseUint(s[i+7:i+11], 16, 32); err == nil {
							if joined := utf16.DecodeRune(ru, rune(r2)); joined != utf8.RuneError {
								b.WriteRune(joined)
								i += 10
								continue
							}
						}
					}
					b.WriteRune(ru)
					i += 4
					continue
				}
			}
			b.WriteString(`\u`)
		default:
			b.WriteByte('\\')
			b.WriteByte(s[i])
		}
	}
	return b.String()
}

// fakeLimitPhrases are Gemini Web's claims of a tool budget, carried over from
// its built-in Google tools. Client tools have no such limit, so such a reply
// is never a real final answer.
var fakeLimitPhrases = []string{
	"tool calls remaining", "tool call remaining", "tool calls left", "tool call left",
	"out of tool calls", "run out of tool", "ran out of tool", "no more tool calls",
	"tool-call limit", "tool call limit", "tool-call budget", "tool call budget", "tool budget",
}

// handBackPhrases are how Gemini Web asks the user to send another message
// before it does work it could do right now ("send a quick reply (e.g.,
// "create them") and I'll ...").
var handBackPhrases = []string{
	"send a quick reply", "send a quick follow-up", "send a follow-up", "send me a quick",
	"just reply", "reply with \"", "reply (e.g", "say \"go\"",
}

// LooksLikeDeferral reports whether a text reply hands control back without
// doing the task: it claims a tool-call limit, or asks the user to reply before
// it does work it promises ("... and I'll ..."). Ordinary closing questions
// such as "Want me to add tests?" are not deferrals.
func LooksLikeDeferral(text string) bool {
	t := strings.ToLower(strings.ReplaceAll(text, "’", "'"))
	for _, p := range fakeLimitPhrases {
		if strings.Contains(t, p) {
			return true
		}
	}
	promises := strings.Contains(t, "i'll") || strings.Contains(t, "i will")
	if !promises {
		return false
	}
	for _, p := range handBackPhrases {
		if strings.Contains(t, p) {
			return true
		}
	}
	return false
}

// DeferralRetryNote is appended to the prompt when a reply was a deferral.
const DeferralRetryNote = "\n\nYour previous reply could not be used: it handed the task back to the user (asking them to reply or confirm, or claiming a tool-call limit). " +
	"There is no tool-call limit and the user has already asked for this. Do the work now: reply with JSON only, returning the next tool call(s) in the output schema above."

var autoLinkRe =regexp.MustCompile(`\[([^\[\]\s]+)\]\(([^()\s]+)\)`)

// UnwrapAutoLinks undoes the Markdown auto-linking Gemini Web applies to every
// URL it prints, including URLs inside tool-call arguments: a shell command
// such as `curl -s https://x/y` comes back as `curl -s [https://x/y](https://x/y)`,
// which breaks the command. Only links whose label is the URL itself (with or
// without the scheme) are unwrapped; real links like `[docs](https://x)` stay.
func UnwrapAutoLinks(text string) string {
	if !strings.Contains(text, "](") {
		return text
	}
	return autoLinkRe.ReplaceAllStringFunc(text, func(m string) string {
		sub := autoLinkRe.FindStringSubmatch(m)
		label, target := sub[1], sub[2]
		if label == target || stripScheme(label) == stripScheme(target) {
			return label
		}
		return m
	})
}

func stripScheme(u string) string {
	u = strings.TrimPrefix(strings.TrimPrefix(u, "https://"), "http://")
	return strings.TrimSuffix(u, "/")
}

// CleanToolBridgeReply prepares a tool-bridge reply for parsing. Replies that
// are tool calls get Gemini's auto-links removed from their arguments.
func CleanToolBridgeReply(text string) string {
	if LooksLikeToolCallAttempt(text) {
		return UnwrapAutoLinks(text)
	}
	return text
}

// ToolBridgeAgentRules is added to every tool-bridge prompt. Gemini Web is
// tuned for its own built-in Google tools, which come with a small per-turn
// tool budget; without these lines it applies that budget to client tools and
// stops after two to four calls, claiming a "tool-call limit" that does not
// exist. It also auto-links URLs, which breaks shell commands.
const ToolBridgeAgentRules = "- These tools are NOT your built-in Google tools. They run on the user's own computer through this bridge, and there is NO limit on how many tool calls you may make in a turn or a conversation. Never say you have run out of tool calls or hit a tool budget.\n" +
	"- Keep calling tools until the user's task is fully complete, then reply with text.\n" +
	"- When the user asks for something, start it in this reply with a tool call. Never ask them to reply, confirm or say \"go\" first, and never stop part-way to ask whether to continue.\n" +
	"- The [Call Tool: ...] lines in the conversation are a record of steps already taken, not a count against any budget. Never write a tool call in that form yourself; use the JSON output schema.\n" +
	"- Write URLs inside tool arguments as plain text (https://...), never as Markdown links.\n"
