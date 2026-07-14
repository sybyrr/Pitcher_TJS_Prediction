import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

const root = new URL("../", import.meta.url);

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", { headers: { accept: "text/html" } }),
    {
      ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) },
    },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("server-renders the final research dashboard shell", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<html lang="ko">/i);
  assert.match(html, /<title>PAINS \| MLB TJS 위험 순위 연구 대시보드<\/title>/i);
  assert.match(html, /MLB TJS · RETROSPECTIVE DEMO/);
  assert.match(html, /연구 경보 ≠ 임상 진단/);
  assert.match(html, /현재 선수의 실시간 위험 명단이 아닙니다/);
  assert.match(html, /불펜 강제 quota는 채택하지 않았습니다/);
  assert.match(html, /game_date &lt; t/);
  assert.match(html, /e14ba800227a5b65a12ca55114e106e20a4636857ef947d5997b9e496e02fac8/);
  assert.doesNotMatch(html, /codex-preview|Your site is taking shape|react-loading-skeleton/);
});

test("locks the demo asset and removes starter-only dependencies", async () => {
  const [source, deployed, socialAsset, manifestText, page, packageJson] = await Promise.all([
    readFile(new URL("../results/phase3/demo_test_top20.csv", root)),
    readFile(new URL("public/data/demo_test_top20.csv", root)),
    readFile(new URL("public/og-research.png", root)),
    readFile(new URL("public/data/manifest.json", root), "utf8"),
    readFile(new URL("app/page.tsx", root), "utf8"),
    readFile(new URL("package.json", root), "utf8"),
  ]);

  assert.deepEqual(deployed, source);
  const digest = createHash("sha256").update(deployed).digest("hex");
  assert.equal(digest, "d2f68cd38bbd5cdbb9bd5009280f5afaf534db0eea494312e95c4c083ade1228");
  const manifest = JSON.parse(manifestText);
  assert.equal(manifest.data_sha256, digest);
  assert.equal(manifest.model_state_sha256, "e14ba800227a5b65a12ca55114e106e20a4636857ef947d5997b9e496e02fac8");
  assert.equal(
    createHash("sha256").update(socialAsset).digest("hex"),
    manifest.social_asset_sha256,
  );
  assert.equal(manifest.prospective, false);

  assert.match(page, /fetch\("\/data\/demo_test_top20\.csv"\)/);
  assert.match(page, /surgery_within_150d/);
  assert.match(page, /과거 재현 검증에만 표시/);
  assert.doesNotMatch(packageJson, /react-loading-skeleton/);
  await assert.rejects(access(new URL("app/_sites-preview/SkeletonPreview.tsx", root)));
});
