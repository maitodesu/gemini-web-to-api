package providers

import (
	"testing"
	"time"
)

// Trimmed from a real UsagePageQuery reply (first part plus a deferred part).
const usageReply = `[{"data":{"viewer":{"rpcs":{"getUserStatus":{"featureGates":["CAN_SEE_USAGE_LIMITS"]},
"getUsageInfo":{"subscriptionTier":"TIER1","pools":[
{"poolType":"GXU_WEEKLY","percentageUsed":0.10307323,"gxu":{"windowRefreshesAt":"2026-11-21T18:31:12.920289Z"}},
{"poolType":"GXU_FIVE_HOURLY","percentageUsed":0.01,"gxu":{"windowRefreshesAt":"2026-11-15T19:31:12.920166Z"}}],
"advancedSettings":null,"useOverageAiCredits":false},"id":"ViewerRpcs__viewer_rpcs_v1"}}},"path":[]},
{"data":{"__typename":"Viewer"},"path":["viewer","__defer__:3:x"]}]`

func TestParseUsage(t *testing.T) {
	now := time.Date(2026, 11, 15, 18, 0, 0, 0, time.UTC)
	u, err := parseUsage([]byte(usageReply), now)
	if err != nil {
		t.Fatal(err)
	}
	if u.SubscriptionTier != "TIER1" || len(u.Pools) != 2 {
		t.Fatalf("got %+v", u)
	}
	w := u.Pools[0]
	if w.Window != "weekly" || w.PercentUsed < 10.3 || w.PercentUsed > 10.31 || w.ResetsAt.Day() != 21 {
		t.Fatalf("weekly pool %+v", w)
	}
	if u.Pools[1].Window != "5h" || u.Pools[1].PercentUsed != 1 {
		t.Fatalf("5h pool %+v", u.Pools[1])
	}
}

func TestParseUsageErrors(t *testing.T) {
	if _, err := parseUsage([]byte(`[{"data":null,"errors":[{"message":"unknown query"}]}]`), time.Now()); err == nil {
		t.Fatal("want error for GraphQL errors")
	}
	if _, err := parseUsage([]byte(`<html>sign in</html>`), time.Now()); err == nil {
		t.Fatal("want error for HTML")
	}
}

func TestNewUsageCacheDefaultID(t *testing.T) {
	if newUsageCache("").queryID != defaultUsageQueryID || newUsageCache(" x ").queryID != "x" {
		t.Fatal("query ID override")
	}
}
