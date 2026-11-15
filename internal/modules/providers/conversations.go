package providers

import (
	"crypto/sha256"
	"encoding/hex"
	"strings"
	"sync"
	"time"

	"gemini-web-to-api/internal/commons/models"
)

// Session reuse (GEMINI_SESSION_REUSE).
//
// OpenAI- and Claude-style clients are stateless: every request carries the
// whole conversation. Without reuse each request becomes a new Gemini Web chat
// that re-reads all of it. With reuse the server remembers, for every reply it
// gave, a fingerprint of the messages that led to it plus the Gemini chat IDs
// of that reply. When a later request starts with those same messages followed
// by the client's copy of the reply, only the messages after it are sent, into
// the same Gemini chat. Anything else (an edited message, a regenerate, a
// compacted history, a restart) finds no match and starts a new chat.

const (
	conversationTTL        = 6 * time.Hour
	conversationMaxEntries = 4096
)

// ConversationStore maps conversation fingerprints to Gemini chat positions.
type ConversationStore struct {
	mu      sync.Mutex
	entries map[string]conversationEntry
	now     func() time.Time
}

type conversationEntry struct {
	meta    SessionMetadata
	expires time.Time
}

func newConversationStore(enabled bool) *ConversationStore {
	if !enabled {
		return nil
	}
	return &ConversationStore{entries: make(map[string]conversationEntry), now: time.Now}
}

// ConversationPlan says how to send one request: into a new chat, or as a
// continuation of a known one.
type ConversationPlan struct {
	store *ConversationStore
	key   string // fingerprint of all the request's messages
	// Meta is the Gemini chat position to continue; nil means a new chat.
	Meta *SessionMetadata
	// NewFrom is the index of the first message to send when continuing.
	NewFrom int
}

// Continues reports whether the request goes into an existing Gemini chat.
func (p *ConversationPlan) Continues() bool { return p != nil && p.Meta != nil }

// Option returns the generate option for the plan (nil for a new chat).
func (p *ConversationPlan) Option() GenerateOption {
	if !p.Continues() {
		return nil
	}
	meta := *p.Meta
	return func(c *GenerateConfig) { c.Conversation = &meta }
}

// Fresh returns a plan that starts a new chat but still records the reply, for
// retrying a continuation that failed.
func (p *ConversationPlan) Fresh() *ConversationPlan {
	if p == nil {
		return nil
	}
	return &ConversationPlan{store: p.store, key: p.key}
}

// Save records the Gemini chat position of resp as the reply to the request's
// messages, so the next request of the conversation can continue it.
func (p *ConversationPlan) Save(resp *Response) {
	if p == nil || p.store == nil || resp == nil {
		return
	}
	meta, ok := sessionFromResponse(resp)
	if !ok {
		return
	}
	p.store.put(p.key, meta)
}

// PlanConversation decides whether msgs continue a conversation this server
// already answered. scope separates conversations that must not share a chat
// (API route, model, system prompt). With reuse off the plan is always a new
// chat and Save does nothing.
func (c *Client) PlanConversation(scope string, msgs []models.Message) *ConversationPlan {
	if c == nil || c.conversations == nil || len(msgs) == 0 {
		return &ConversationPlan{}
	}
	return c.conversations.plan(scope, msgs)
}

func (s *ConversationStore) plan(scope string, msgs []models.Message) *ConversationPlan {
	keys := conversationKeys(scope, msgs)
	p := &ConversationPlan{store: s, key: keys[len(msgs)]}

	s.mu.Lock()
	defer s.mu.Unlock()
	now := s.now()
	// Longest known prefix first. msgs[k] must be the client's copy of the
	// reply to msgs[:k], and something new must follow it.
	for k := len(msgs) - 2; k >= 1; k-- {
		e, ok := s.entries[keys[k]]
		if !ok || now.After(e.expires) || !isAssistantRole(msgs[k].Role) {
			continue
		}
		meta := e.meta
		p.Meta = &meta
		p.NewFrom = k + 1
		break
	}
	return p
}

func (s *ConversationStore) put(key string, meta SessionMetadata) {
	s.mu.Lock()
	defer s.mu.Unlock()
	now := s.now()
	if len(s.entries) >= conversationMaxEntries {
		for k, e := range s.entries {
			if now.After(e.expires) {
				delete(s.entries, k)
			}
		}
		// Still full: drop the entries closest to expiry.
		for len(s.entries) >= conversationMaxEntries {
			var oldest string
			var oldestAt time.Time
			for k, e := range s.entries {
				if oldest == "" || e.expires.Before(oldestAt) {
					oldest, oldestAt = k, e.expires
				}
			}
			delete(s.entries, oldest)
		}
	}
	s.entries[key] = conversationEntry{meta: meta, expires: now.Add(conversationTTL)}
}

// conversationKeys returns the fingerprint of msgs[:k] for k = 0..len(msgs).
// Each key chains the previous one, so all prefixes cost one pass.
func conversationKeys(scope string, msgs []models.Message) []string {
	keys := make([]string, len(msgs)+1)
	h := sha256.Sum256([]byte("conversation\x00" + scope))
	keys[0] = hex.EncodeToString(h[:])
	for i, m := range msgs {
		d := sha256.New()
		d.Write(h[:])
		d.Write([]byte(normalizedRole(m.Role)))
		d.Write([]byte{0})
		d.Write([]byte(strings.TrimSpace(m.Content)))
		for _, a := range m.Attachments {
			d.Write([]byte{0})
			d.Write([]byte(a.MimeType))
			d.Write([]byte{0})
			d.Write([]byte(a.Data))
		}
		copy(h[:], d.Sum(nil))
		keys[i+1] = hex.EncodeToString(h[:])
	}
	return keys
}

func normalizedRole(role string) string {
	switch r := strings.ToLower(strings.TrimSpace(role)); r {
	case "assistant", "model":
		return "assistant"
	case "":
		return "user"
	default:
		return r
	}
}

func isAssistantRole(role string) bool { return normalizedRole(role) == "assistant" }

// sessionFromResponse reads the chat position of a reply. All three IDs are
// needed: Gemini continues a chat from a specific reply candidate.
func sessionFromResponse(resp *Response) (SessionMetadata, bool) {
	get := func(k string) string {
		v, _ := resp.Metadata[k].(string)
		return v
	}
	meta := SessionMetadata{ConversationID: get("cid"), ResponseID: get("rid"), ChoiceID: get("rcid")}
	if meta.ConversationID == "" {
		meta.ConversationID = resp.ConversationID
	}
	if meta.ConversationID == "" || meta.ResponseID == "" || meta.ChoiceID == "" {
		return SessionMetadata{}, false
	}
	return meta, true
}

// conversationMetadata is the StreamGenerate chat-metadata field that
// continues a chat after the given reply.
func conversationMetadata(m *SessionMetadata) []interface{} {
	return []interface{}{m.ConversationID, m.ResponseID, m.ChoiceID, nil, nil, nil, nil, nil, nil, ""}
}

// chatIDsFromPayload reads the conversation and response IDs of a
// StreamGenerate frame. They arrive as payload[1] = ["c_...", "r_..."], or in
// older replies as a bare conversation ID string.
func chatIDsFromPayload(payload []interface{}) (cid, rid string) {
	if len(payload) < 2 {
		return "", ""
	}
	switch v := payload[1].(type) {
	case string:
		return v, ""
	case []interface{}:
		if len(v) > 0 {
			cid, _ = v[0].(string)
		}
		if len(v) > 1 {
			rid, _ = v[1].(string)
		}
	}
	return cid, rid
}

// SetSessionReuse turns session reuse on or off, for tests of other packages.
func (c *Client) SetSessionReuse(enabled bool) {
	c.conversations = newConversationStore(enabled)
}
