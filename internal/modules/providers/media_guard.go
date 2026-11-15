package providers

import (
	"context"
	"errors"
	"regexp"
	"strings"

	"go.uber.org/zap"
)

// ErrUnrequestedMedia means Gemini Web started its own video or music tool
// for a text request. Gemini does this when a long pasted transcript looks
// like a media request, then answers with "I'm generating your video…",
// which an API client reads as a real reply.
var ErrUnrequestedMedia = errors.New("Gemini started generating media for a text-only request")

// TextOnlyNote is added to the prompt when a reply has to be asked for again
// because Gemini started a media tool.
const TextOnlyNote = "\n\nNote: this is a text-only API. Answer in text, or in the JSON format given above if there is one. " +
	"Do not create videos, music or images, do not use Gemini's own media tools, and never say you are generating one."

// MediaUnavailableReply replaces a reply that still starts media generation
// after TextOnlyNote.
const MediaUnavailableReply = "I can't create videos or music through this chat API, so nothing was generated. " +
	"Video generation is available separately through the bridge's /v1/videos endpoint."

// WithTextOnly marks a request whose caller can only use text. A reply that
// starts Gemini's video or music generation is then never returned as is.
func WithTextOnly() GenerateOption {
	return func(c *GenerateConfig) {
		c.TextOnly = true
	}
}

// mediaAckPatterns match the short acknowledgements Gemini Web writes when it
// starts a video or music job from a chat. The finished card only arrives
// later in the saved conversation, so the text is all a chat reply carries.
var mediaAckPatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?i)\b(i'?m|i am|i'?ll|i will|i'?ve started|now|currently)\s+(generat(e|ing)|creat(e|ing)|mak(e|ing)|produc(e|ing)|render(ing)?|compos(e|ing))\b[^.\n]{0,40}\b(video|videos|music|song|track|clip)\b`),
	regexp.MustCompile(`(?i)\b(generating|rendering)\s+(your|a|the|this)\s+(video|music|song|track|clip)\b`),
	regexp.MustCompile(`(?i)\b(your|the)\s+(video|song|track|clip)\s+(is|will be)\s+(being generated|being created|ready (soon|shortly|in)|on (its|the) way)`),
}

// mediaAckMaxLen bounds the replies checked for an acknowledgement: Gemini's
// are a sentence or two, while a long answer that mentions videos is content.
const mediaAckMaxLen = 600

// StartedMediaGeneration reports whether a reply is Gemini starting its own
// video or music generation rather than answering.
func StartedMediaGeneration(resp *Response) bool {
	if resp == nil {
		return false
	}
	if len(resp.Videos) > 0 || len(resp.Audios) > 0 {
		return true
	}
	text := strings.TrimSpace(resp.Text)
	if text == "" || len(text) > mediaAckMaxLen {
		return false
	}
	for _, re := range mediaAckPatterns {
		if re.MatchString(text) {
			return true
		}
	}
	return false
}

// guardTextOnly applies WithTextOnly to a finished reply. A continued chat
// returns ErrUnrequestedMedia so the caller restarts it from the full history;
// a fresh request is asked once more with TextOnlyNote.
func (c *Client) guardTextOnly(ctx context.Context, prompt string, resp *Response, options []GenerateOption) (*Response, error) {
	config := &GenerateConfig{}
	for _, opt := range options {
		opt(config)
	}
	if !config.TextOnly || !StartedMediaGeneration(resp) {
		return resp, nil
	}
	if config.Conversation != nil {
		c.log.Warn("Gemini started media generation in a continued chat; restarting the chat")
		return nil, ErrUnrequestedMedia
	}
	c.log.Warn("Gemini started media generation for a text request; asking again for text")
	retry, err := c.generateContent(ctx, prompt+TextOnlyNote, nil, options...)
	if err != nil {
		return nil, err
	}
	if StartedMediaGeneration(retry) {
		// A reply, not an error: clients and proxies retry errors, and every
		// retry would start another generation on the account.
		c.log.Warn("Gemini started media generation again; replying that media is unavailable", zap.Int("attempts", 2))
		retry.Text = MediaUnavailableReply
		retry.Videos, retry.Audios = nil, nil
	}
	return retry, nil
}

// isChatNotContinuableError reports Gemini's [13 1097] reply, which it gives
// for a chat it will no longer continue.
func isChatNotContinuableError(err error) bool {
	return err != nil && strings.Contains(err.Error(), "[13 1097]")
}
