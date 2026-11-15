package providers

import (
	"testing"
	"time"

	"gemini-web-to-api/internal/commons/models"
)

func reply(cid, rid, rcid string) *Response {
	return &Response{Metadata: map[string]any{"cid": cid, "rid": rid, "rcid": rcid}}
}

func TestConversationPlanContinuesKnownPrefix(t *testing.T) {
	c := &Client{}
	c.SetSessionReuse(true)

	turn1 := []models.Message{{Role: "system", Content: "be brief"}, {Role: "user", Content: "hi"}}
	p := c.PlanConversation("openai|m", turn1)
	if p.Continues() {
		t.Fatal("first turn must start a new chat")
	}
	p.Save(reply("c_1", "r_1", "rc_1"))

	turn2 := append(append([]models.Message{}, turn1...),
		models.Message{Role: "assistant", Content: "hello"}, models.Message{Role: "user", Content: "and?"})
	p = c.PlanConversation("openai|m", turn2)
	if !p.Continues() || p.NewFrom != 3 || p.Meta.ConversationID != "c_1" || p.Meta.ChoiceID != "rc_1" {
		t.Fatalf("plan = %+v, want continuation from message 3 of c_1", p)
	}
	p.Save(reply("c_1", "r_2", "rc_2"))

	turn3 := append(append([]models.Message{}, turn2...),
		models.Message{Role: "assistant", Content: "more"}, models.Message{Role: "user", Content: "ok"})
	if p = c.PlanConversation("openai|m", turn3); !p.Continues() || p.NewFrom != 5 || p.Meta.ResponseID != "r_2" {
		t.Fatalf("plan = %+v, want continuation after r_2", p)
	}

	// Another model, an edited message or a regenerate start a new chat.
	if c.PlanConversation("openai|other", turn2).Continues() {
		t.Fatal("a different scope must not share the chat")
	}
	edited := append([]models.Message{}, turn2...)
	edited[1].Content = "hey"
	if c.PlanConversation("openai|m", edited).Continues() {
		t.Fatal("an edited history must start a new chat")
	}
	if c.PlanConversation("openai|m", turn1).Continues() {
		t.Fatal("re-sending the same messages (regenerate) must start a new chat")
	}
}

func TestConversationPlanOffAndExpiry(t *testing.T) {
	off := &Client{}
	msgs := []models.Message{{Role: "user", Content: "hi"}}
	p := off.PlanConversation("s", msgs)
	p.Save(reply("c", "r", "rc")) // no-op, must not panic
	var nilClient *Client
	if nilClient.PlanConversation("s", msgs).Continues() {
		t.Fatal("nil client must not continue")
	}

	c := &Client{}
	c.SetSessionReuse(true)
	now := time.Now()
	c.conversations.now = func() time.Time { return now }
	c.PlanConversation("s", msgs).Save(reply("c", "r", "rc"))
	next := append(append([]models.Message{}, msgs...), models.Message{Role: "assistant", Content: "a"}, models.Message{Role: "user", Content: "b"})
	if !c.PlanConversation("s", next).Continues() {
		t.Fatal("want continuation before expiry")
	}
	now = now.Add(conversationTTL + time.Minute)
	if c.PlanConversation("s", next).Continues() {
		t.Fatal("expired entry must not be continued")
	}

	// A reply without all three IDs cannot be continued.
	c.PlanConversation("t", msgs).Save(reply("c", "", "rc"))
	if c.PlanConversation("t", next).Continues() {
		t.Fatal("incomplete IDs must not be stored")
	}
}

func TestChatIDsFromPayload(t *testing.T) {
	cid, rid := chatIDsFromPayload([]interface{}{nil, []interface{}{"c_1", "r_1"}})
	if cid != "c_1" || rid != "r_1" {
		t.Fatalf("got %q %q", cid, rid)
	}
	if cid, rid = chatIDsFromPayload([]interface{}{nil, "c_2"}); cid != "c_2" || rid != "" {
		t.Fatalf("got %q %q", cid, rid)
	}
}
