// Records the user-guide videos by driving a deployed (or local) instance.
//
// Usage:
//   APP_URL=https://... APP_PASSWORD=word-word-word node record.mjs [flow ...]
//
// With no args, records all flows in order. Pass flow names (e.g. "03-click")
// to re-record individual ones. Raw .webm files land in ./raw/; post-process
// with ../render-user-guide-videos.sh.
//
// Flows after 01-login reuse the saved storage state (raw/state.json) so each
// video starts inside the app rather than at the password gate.

import { chromium } from "playwright";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, "../..");
const RAW = path.join(HERE, "raw");
const STATE = path.join(RAW, "state.json");
const FIXTURE = process.env.FIXTURE ?? path.join(REPO, "tests/fixtures/harness/sample.mp4");

const APP_URL = process.env.APP_URL;
const PASSWORD = process.env.APP_PASSWORD;
if (!APP_URL || !PASSWORD) {
  console.error("Set APP_URL and APP_PASSWORD (hint: ./deploy/deploy.sh password)");
  process.exit(1);
}

const VIEWPORT = { width: 1280, height: 800 };
// Video pixel coords from the harness fixture (tests/fixtures/harness/):
// pressure gauge click point and portafilter box on sample.mp4 (360x640).
const GAUGE = { x: 160, y: 296 };
const PORTAFILTER = { x1: 168, y1: 345, x2: 225, y2: 460 };

const pause = (ms) => new Promise((r) => setTimeout(r, ms));

/** Map a video-pixel coordinate to screen coordinates via the canvas bbox.
 *  The canvas drawing buffer is the natural video size and the element is
 *  CSS-transformed (pan+scale), so its bounding box IS the on-screen image. */
async function canvasPoint(page, vx, vy) {
  const canvas = page.locator("canvas").first();
  const box = await canvas.boundingBox();
  const dims = await canvas.evaluate((el) => ({ w: el.width, h: el.height }));
  return {
    x: box.x + (vx / dims.w) * box.width,
    y: box.y + (vy / dims.h) * box.height,
  };
}

async function waitForAnnotationUi(page, timeout = 300_000) {
  await page.locator("canvas").first().waitFor({ state: "visible", timeout });
  await pause(2000); // let frames/masks render for the recording
}

async function createClass(page, name) {
  const first = page.getByRole("button", { name: "Create first class" });
  if (await first.isVisible().catch(() => false)) {
    await first.click();
  } else {
    await page.getByRole("button", { name: /New class/ }).click();
  }
  const input = page.locator('input[placeholder="Class name"]');
  await input.fill(name);
  await pause(400);
  await input.press("Enter");
  await pause(800);
}

const FLOWS = {
  "01-login": {
    useState: false,
    run: async (page, context) => {
      await page.goto(APP_URL, { timeout: 180_000 });
      const input = page.getByPlaceholder("three-word-password");
      await input.waitFor({ timeout: 180_000 }); // cold start can take ~90s
      await pause(1000);
      await input.pressSequentially("wrong-password-demo", { delay: 60 });
      await input.press("Enter");
      await page.getByText("Wrong password").waitFor({ timeout: 30_000 });
      await pause(1800);
      await input.fill("");
      await input.pressSequentially(PASSWORD, { delay: 60 });
      await input.press("Enter");
      await page.getByRole("tab", { name: "Recent" }).waitFor({ timeout: 60_000 });
      await pause(2000);
      await context.storageState({ path: STATE });
    },
  },

  "02-upload": {
    run: async (page) => {
      await page.goto(APP_URL, { timeout: 180_000 });
      await page.getByRole("tab", { name: "New Video" }).waitFor({ timeout: 120_000 });
      await pause(1000);
      await page.getByRole("tab", { name: "New Video" }).click();
      await pause(800);
      await page.locator('input[type="file"]').setInputFiles(FIXTURE);
      await pause(1200);
      await page.getByRole("button", { name: "Upload" }).click();
      // extraction + SAM3 init stream progress; UI lands on the canvas
      await waitForAnnotationUi(page, 360_000);
      await pause(2000);
    },
  },

  "03-click": {
    run: async (page) => {
      await page.goto(APP_URL, { timeout: 180_000 });
      await waitForAnnotationUi(page); // auto-resumes the live session
      await createClass(page, "pressure gauge");
      await page.getByRole("button", { name: "Click", exact: true }).click();
      await pause(800);
      const p = await canvasPoint(page, GAUGE.x, GAUGE.y);
      await page.mouse.move(p.x, p.y, { steps: 15 });
      await pause(400);
      await page.mouse.click(p.x, p.y);
      await pause(4000); // SAM3 inference + mask render
      await page.getByRole("button", { name: "Select", exact: true }).click();
      await pause(1500);
    },
  },

  "04-box": {
    run: async (page) => {
      await page.goto(APP_URL, { timeout: 180_000 });
      await waitForAnnotationUi(page);
      await createClass(page, "portafilter");
      await page.getByRole("button", { name: "Box", exact: true }).click();
      await pause(800);
      const a = await canvasPoint(page, PORTAFILTER.x1, PORTAFILTER.y1);
      const b = await canvasPoint(page, PORTAFILTER.x2, PORTAFILTER.y2);
      await page.mouse.move(a.x, a.y, { steps: 15 });
      await page.mouse.down();
      await page.mouse.move(b.x, b.y, { steps: 30 });
      await page.mouse.up();
      await pause(4000);
      await page.getByRole("button", { name: "Select", exact: true }).click();
      await pause(1500);
    },
  },

  "05-detect": {
    run: async (page) => {
      await page.goto(APP_URL, { timeout: 180_000 });
      await waitForAnnotationUi(page);
      // select the "pressure gauge" class so Detect prefills its name
      await page.getByText("pressure gauge", { exact: true }).first().click();
      await pause(600);
      await page.getByRole("button", { name: "Detect", exact: true }).click();
      const input = page.locator('input[placeholder*="Describe object"]');
      await input.waitFor({ timeout: 10_000 });
      await input.fill("");
      await input.pressSequentially("pressure gauge", { delay: 50 });
      await pause(400);
      await page.getByRole("button", { name: "Find", exact: true }).click();
      await pause(8000); // text detection inference + mask render
      // Delete the detected object so the session keeps one object per
      // class (text detections aren't covered by Undo). The new object has
      // the highest id; its row contains a nested X delete button.
      const texts = await page.getByRole("button", { name: /Object #\d+/ }).allTextContents();
      const maxId = Math.max(...texts.map((t) => parseInt(t.match(/#(\d+)/)[1], 10)));
      const row = page.getByRole("button", { name: new RegExp(`Object #${maxId}(\\D|$)`) });
      await row.locator("button").click();
      await pause(800);
      await page.getByRole("dialog").getByRole("button", { name: "Delete", exact: true }).click();
      await pause(2000);
    },
  },

  "06-propagate": {
    run: async (page) => {
      await page.goto(APP_URL, { timeout: 180_000 });
      await waitForAnnotationUi(page);
      await page.getByRole("button", { name: /Object #1/ }).click();
      await pause(800);
      await page.getByRole("button", { name: "Forward", exact: true }).click();
      await page.getByText(/Propagation complete/).waitFor({ timeout: 300_000 });
      await pause(2500);
    },
  },

  "07-propagate-multi": {
    run: async (page) => {
      await page.goto(APP_URL, { timeout: 180_000 });
      await waitForAnnotationUi(page);
      await page.getByRole("button", { name: /Object #1/ }).click();
      await pause(500);
      await page.getByRole("button", { name: /Object #2/ }).click({ modifiers: ["Shift"] });
      await pause(1200); // chips render in the propagation bar
      await page.getByRole("button", { name: "Both", exact: true }).click();
      await page.getByText(/Propagation complete/).waitFor({ timeout: 300_000 });
      await pause(2500);
    },
  },

  "08-export": {
    run: async (page) => {
      await page.goto(APP_URL, { timeout: 180_000 });
      await waitForAnnotationUi(page);
      await page.getByRole("button", { name: "Export Annotations" }).click();
      await pause(1200);
      const download = page.waitForEvent("download", { timeout: 120_000 });
      await page.getByRole("button", { name: "Export", exact: true }).click();
      await download;
      await pause(2000);
    },
  },

  "09-close-resume": {
    run: async (page) => {
      await page.goto(APP_URL, { timeout: 180_000 });
      await waitForAnnotationUi(page);
      await page.locator("aside button:has(svg.lucide-x)").first().click();
      await page.getByRole("button", { name: "Close Session" }).click();
      await page.getByRole("tab", { name: "Recent" }).waitFor({ timeout: 120_000 });
      await pause(1500);
      await page.getByRole("button", { name: /sample\.mp4/ }).first().click();
      await waitForAnnotationUi(page, 300_000); // GCS download + SAM3 re-init
      await pause(2000);
    },
  },
};

const requested = process.argv.slice(2);
const names = requested.length ? requested : Object.keys(FLOWS);
for (const n of names) {
  if (!FLOWS[n]) {
    console.error(`Unknown flow "${n}". Available: ${Object.keys(FLOWS).join(", ")}`);
    process.exit(1);
  }
}

fs.mkdirSync(RAW, { recursive: true });
const browser = await chromium.launch();

for (const name of names) {
  const flow = FLOWS[name];
  console.log(`▶ recording ${name} ...`);
  const context = await browser.newContext({
    viewport: VIEWPORT,
    recordVideo: { dir: RAW, size: VIEWPORT },
    ...(flow.useState === false ? {} : { storageState: STATE }),
  });
  const page = await context.newPage();
  try {
    await flow.run(page, context);
    const video = page.video();
    await context.close(); // flushes the recording
    const out = path.join(RAW, `${name}.webm`);
    fs.rmSync(out, { force: true });
    fs.renameSync(await video.path(), out);
    console.log(`✔ ${name} → raw/${name}.webm`);
  } catch (err) {
    await page.screenshot({ path: path.join(RAW, `${name}-FAILED.png`) }).catch(() => {});
    await context.close().catch(() => {});
    console.error(`✘ ${name} failed:`, err.message);
    process.exit(1);
  }
}

await browser.close();
console.log("All flows recorded.");
