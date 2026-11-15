package providers

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"
)

// Gemini Web shows the account's usage limits at gemini.google.com/usage. The
// page reads them with the persisted GraphQL query UsagePageQuery, a plain GET
// with the session cookies (queries need no XSRF token). Google can rotate the
// query ID with a web release; GEMINI_USAGE_QUERY_ID overrides it.
const (
	defaultUsageQueryID = "2/r9A7h8PGYFROeibFhACp9h3mHf72UWS3JhgYaQYlPoo="
	geminiGraphQLQuery  = "https://gemini.google.com/_/BardChatUi/graphql/schemas/GEMINI_WEB_GRAPHQL/executeQuery"
	usageCacheTTL       = 60 * time.Second
	maxUsageResponse    = 1 << 20
)

// usageQueryVariables are the variables the /usage page sends with the query.
const usageQueryVariables = `{"g1Request":{"entrypoints":["GEMINI_USAGE_SETTINGS_UPGRADE_BANNER","GEMINI_USAGE_SETTINGS_AI_CREDITS_UPSELL_CARD","GEMINI_USAGE_SETTINGS_UPGRADE_AND_CREDITS_BANNER"]},"workspaceRequest":{"recommendationsForSurfaceRequests":[{"recommendationSurface":{"rpOnramp":"GEMINI_CHAT_USAGE_LIMITS_UPSELL","version":"1"},"maxRecommendationCount":1}]}}`

// UsagePool is one usage window, e.g. the five-hour or the weekly budget.
type UsagePool struct {
	Type        string    `json:"type"`         // GXU_FIVE_HOURLY, GXU_WEEKLY, ...
	Window      string    `json:"window"`       // "5h", "weekly", or the raw type
	PercentUsed float64   `json:"percent_used"` // 0-100
	ResetsAt    time.Time `json:"resets_at"`
}

// Usage is the account's Gemini Web usage as shown on the Usage limits page.
type Usage struct {
	SubscriptionTier string      `json:"subscription_tier"` // TIER1 = Google AI Pro, TIER2 = Ultra
	Pools            []UsagePool `json:"pools"`
	OverageCredits   bool        `json:"use_overage_ai_credits"`
	FetchedAt        time.Time   `json:"fetched_at"`
}

type usageCache struct {
	mu      sync.Mutex
	queryID string
	at      time.Time
	data    *Usage
}

// newUsageCache uses queryID (GEMINI_USAGE_QUERY_ID) or the built-in ID.
func newUsageCache(queryID string) *usageCache {
	if queryID = strings.TrimSpace(queryID); queryID == "" {
		queryID = defaultUsageQueryID
	}
	return &usageCache{queryID: queryID}
}

// GetUsage returns the account's usage limits, cached for a minute so callers
// polling it do not turn into a stream of requests to Google.
func (c *Client) GetUsage(ctx context.Context) (*Usage, error) {
	if c.usage == nil {
		c.usage = newUsageCache("")
	}
	c.usage.mu.Lock()
	defer c.usage.mu.Unlock()
	if c.usage.data != nil && time.Since(c.usage.at) < usageCacheTTL {
		return c.usage.data, nil
	}
	c.mu.Lock()
	cookieHeader := c.cookieHeader
	authUser := c.authUser
	c.mu.Unlock()
	if cookieHeader == "" {
		return nil, errors.New("client not initialized")
	}
	usage, err := fetchUsage(ctx, http.DefaultClient, c.usage.queryID, cookieHeader, authUser)
	if err != nil {
		return nil, err
	}
	c.usage.data, c.usage.at = usage, time.Now()
	return usage, nil
}

func fetchUsage(ctx context.Context, client *http.Client, queryID, cookieHeader, authUser string) (*Usage, error) {
	query := url.Values{"query_signature": {queryID}, "variables": {usageQueryVariables}}
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, geminiAccountURL(geminiGraphQLQuery, authUser)+"?"+query.Encode(), nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Cookie", cookieHeader)
	req.Header.Set("Accept", "application/json")
	req.Header.Set("Origin", "https://gemini.google.com")
	req.Header.Set("Referer", "https://gemini.google.com/usage")
	req.Header.Set("User-Agent", DefaultHeaders["User-Agent"])
	resp, err := client.Do(req)
	if err != nil {
		var urlErr *url.Error
		if errors.As(err, &urlErr) {
			err = urlErr.Err // the URL carries nothing secret, but keep logs short
		}
		return nil, fmt.Errorf("Gemini usage request failed: %w", err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxUsageResponse))
	if err != nil {
		return nil, fmt.Errorf("read Gemini usage response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Gemini usage request returned HTTP %d (the session may be signed out, or GEMINI_USAGE_QUERY_ID is outdated)", resp.StatusCode)
	}
	return parseUsage(body, time.Now())
}

// parseUsage reads the first payload of the (possibly deferred, multi-part)
// GraphQL reply: data.viewer.rpcs.getUsageInfo.
func parseUsage(body []byte, now time.Time) (*Usage, error) {
	text := strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(string(body)), ")]}'"))
	type usageInfo struct {
		SubscriptionTier string `json:"subscriptionTier"`
		Pools            []struct {
			PoolType       string   `json:"poolType"`
			PercentageUsed *float64 `json:"percentageUsed"`
			Gxu            struct {
				WindowRefreshesAt string `json:"windowRefreshesAt"`
			} `json:"gxu"`
		} `json:"pools"`
		UseOverageAiCredits bool `json:"useOverageAiCredits"`
	}
	type part struct {
		Data struct {
			Viewer struct {
				Rpcs struct {
					GetUsageInfo *usageInfo `json:"getUsageInfo"`
				} `json:"rpcs"`
			} `json:"viewer"`
		} `json:"data"`
		Errors []struct {
			Message string `json:"message"`
		} `json:"errors"`
	}
	var parts []part
	if err := json.Unmarshal([]byte(text), &parts); err != nil {
		var single part
		if err2 := json.Unmarshal([]byte(text), &single); err2 != nil {
			return nil, fmt.Errorf("unexpected Gemini usage response: %w", err)
		}
		parts = []part{single}
	}
	for _, p := range parts {
		info := p.Data.Viewer.Rpcs.GetUsageInfo
		if info == nil {
			if len(p.Errors) > 0 {
				return nil, fmt.Errorf("Gemini usage query failed: %s", p.Errors[0].Message)
			}
			continue
		}
		usage := &Usage{SubscriptionTier: info.SubscriptionTier, OverageCredits: info.UseOverageAiCredits, FetchedAt: now.UTC()}
		for _, pool := range info.Pools {
			up := UsagePool{Type: pool.PoolType, Window: usageWindowName(pool.PoolType)}
			if pool.PercentageUsed != nil {
				up.PercentUsed = *pool.PercentageUsed * 100
			}
			if t, err := time.Parse(time.RFC3339Nano, pool.Gxu.WindowRefreshesAt); err == nil {
				up.ResetsAt = t.UTC()
			}
			usage.Pools = append(usage.Pools, up)
		}
		return usage, nil
	}
	return nil, errors.New("Gemini usage response has no usage info (the account may not have usage limits)")
}

func usageWindowName(poolType string) string {
	switch poolType {
	case "GXU_FIVE_HOURLY":
		return "5h"
	case "GXU_WEEKLY":
		return "weekly"
	case "GXU_DAILY":
		return "daily"
	}
	return strings.ToLower(strings.TrimPrefix(poolType, "GXU_"))
}
