/**
 * IndexedDB frame cache for video annotation sessions.
 *
 * Stores extracted JPEG frames as blobs in IndexedDB, serves them as blob URLs.
 * LRU eviction keeps at most MAX_SESSIONS cached. An in-memory Map avoids
 * repeated IDB reads for the same frame within a page lifetime.
 */

const DB_NAME = "frame-cache";
const DB_VERSION = 1;
const FRAMES_STORE = "frames";
const SESSIONS_STORE = "sessions";
const MAX_SESSIONS = 5;
const BATCH_SIZE = 4;

interface SessionMeta {
  sessionId: string;
  frameCount: number;
  cachedCount: number;
  lastAccessedAt: number;
  totalBytes: number;
}

// ---------------------------------------------------------------------------
// Module-level state
// ---------------------------------------------------------------------------

/** Cached IDB connection — opened once, reused for the lifetime of the page. */
let _db: IDBDatabase | null = null;

/** In-memory blob URL cache: frameKey -> blob URL. Avoids repeated IDB reads. */
const blobUrlCache = new Map<string, string>();

/** Guard so the IDB-unavailable warning is only logged once. */
let _idbWarned = false;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/** Open (or create) the IndexedDB database. Caches the connection in `_db`. */
const openDB = (): Promise<IDBDatabase> => {
  if (_db) return Promise.resolve(_db);

  return new Promise<IDBDatabase>((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(FRAMES_STORE)) {
        const framesStore = db.createObjectStore(FRAMES_STORE);
        framesStore.createIndex("sessionId", "sessionId", { unique: false });
      }
      if (!db.objectStoreNames.contains(SESSIONS_STORE)) {
        db.createObjectStore(SESSIONS_STORE);
      }
    };

    request.onsuccess = () => {
      _db = request.result;

      // If the browser closes the connection behind our back, clear the
      // cached handle so the next call re-opens.
      _db.onclose = () => {
        _db = null;
      };

      resolve(_db);
    };

    request.onerror = () => {
      reject(request.error);
    };
  });
};

/** Deterministic key for a single frame blob. */
const frameKey = (sessionId: string, frameIdx: number): string =>
  `${sessionId}/${frameIdx}`;

/** Write a single frame blob into the frames store. */
const storeFrame = async (
  sessionId: string,
  frameIdx: number,
  blob: Blob,
): Promise<void> => {
  const db = await openDB();
  return new Promise<void>((resolve, reject) => {
    const tx = db.transaction(FRAMES_STORE, "readwrite");
    const store = tx.objectStore(FRAMES_STORE);
    store.put({ blob, sessionId, frameIdx }, frameKey(sessionId, frameIdx));
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
};

/**
 * Read-then-write session metadata so that `totalBytes` accumulates across
 * multiple batches within the same cacheSession run.
 */
const updateSessionMeta = async (
  db: IDBDatabase,
  sessionId: string,
  frameCount: number,
  cachedCount: number,
  totalBytes: number,
): Promise<void> => {
  return new Promise<void>((resolve, reject) => {
    const tx = db.transaction(SESSIONS_STORE, "readwrite");
    const store = tx.objectStore(SESSIONS_STORE);
    const getReq = store.get(sessionId);

    getReq.onsuccess = () => {
      const existing = getReq.result as SessionMeta | undefined;
      const meta: SessionMeta = {
        sessionId,
        frameCount,
        cachedCount,
        lastAccessedAt: Date.now(),
        totalBytes: (existing?.totalBytes ?? 0) + totalBytes,
      };
      store.put(meta, sessionId);
    };

    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
};

/**
 * If we've reached MAX_SESSIONS, evict the least-recently-accessed session
 * to keep disk usage bounded.
 */
const evictIfNeeded = async (db: IDBDatabase): Promise<void> => {
  const sessions = await new Promise<SessionMeta[]>((resolve, reject) => {
    const tx = db.transaction(SESSIONS_STORE, "readonly");
    const store = tx.objectStore(SESSIONS_STORE);
    const req = store.getAll();
    req.onsuccess = () => resolve(req.result as SessionMeta[]);
    req.onerror = () => reject(req.error);
  });

  if (sessions.length < MAX_SESSIONS) return;

  // Find the session with the oldest lastAccessedAt.
  let oldest = sessions[0];
  for (const s of sessions) {
    if (s.lastAccessedAt < oldest.lastAccessedAt) {
      oldest = s;
    }
  }

  await evictSession(oldest.sessionId);
};

// ---------------------------------------------------------------------------
// Exported API
// ---------------------------------------------------------------------------

/**
 * Get a blob URL for a specific frame. Checks the in-memory cache first,
 * then IndexedDB, then falls back to a network fetch (which also populates
 * the cache for next time).
 */
export const getFrame = async (
  sessionId: string,
  frameIdx: number,
): Promise<string> => {
  const key = frameKey(sessionId, frameIdx);

  // 1. In-memory hit — instant.
  const cached = blobUrlCache.get(key);
  if (cached) return cached;

  // 2. Try IndexedDB.
  try {
    const db = await openDB();
    const record = await new Promise<{ blob: Blob } | undefined>(
      (resolve, reject) => {
        const tx = db.transaction(FRAMES_STORE, "readonly");
        const store = tx.objectStore(FRAMES_STORE);
        const req = store.get(key);
        req.onsuccess = () => resolve(req.result as { blob: Blob } | undefined);
        req.onerror = () => reject(req.error);
      },
    );

    if (record) {
      const url = URL.createObjectURL(record.blob);
      blobUrlCache.set(key, url);
      return url;
    }
  } catch {
    if (!_idbWarned) {
      _idbWarned = true;
      console.warn(
        "[frameCache] IndexedDB unavailable — falling back to direct fetch.",
      );
    }
  }

  // 3. IDB miss (or IDB unavailable): fetch from server, cache, return.
  const { fetchFrameBlob } = await import("./api.ts");
  const blob = await fetchFrameBlob(sessionId, frameIdx);

  // Best-effort cache; don't let a storage error break the caller.
  try {
    await storeFrame(sessionId, frameIdx, blob);
  } catch {
    // Storage full or IDB unavailable — still return the blob URL.
  }

  const url = URL.createObjectURL(blob);
  blobUrlCache.set(key, url);
  return url;
};

/**
 * Returns true when every frame for `sessionId` is already in IndexedDB.
 */
export const isSessionCached = async (
  sessionId: string,
): Promise<boolean> => {
  try {
    const db = await openDB();
    const meta = await new Promise<SessionMeta | undefined>(
      (resolve, reject) => {
        const tx = db.transaction(SESSIONS_STORE, "readonly");
        const store = tx.objectStore(SESSIONS_STORE);
        const req = store.get(sessionId);
        req.onsuccess = () => resolve(req.result as SessionMeta | undefined);
        req.onerror = () => reject(req.error);
      },
    );
    if (!meta) return false;
    return meta.cachedCount >= meta.frameCount;
  } catch {
    return false;
  }
};

/**
 * Remove all cached data for a session — frame blobs, session metadata, and
 * revoke any in-memory blob URLs.
 */
export const evictSession = async (sessionId: string): Promise<void> => {
  try {
    const db = await openDB();

    // Delete all frame blobs for this session via the sessionId index.
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(FRAMES_STORE, "readwrite");
      const store = tx.objectStore(FRAMES_STORE);
      const index = store.index("sessionId");
      const cursorReq = index.openKeyCursor(IDBKeyRange.only(sessionId));

      cursorReq.onsuccess = () => {
        const cursor = cursorReq.result;
        if (cursor) {
          store.delete(cursor.primaryKey);
          cursor.continue();
        }
      };

      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });

    // Delete session metadata.
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(SESSIONS_STORE, "readwrite");
      const store = tx.objectStore(SESSIONS_STORE);
      store.delete(sessionId);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  } catch {
    // IDB unavailable — nothing to evict.
  }

  // Revoke and remove in-memory blob URLs for this session.
  const prefix = `${sessionId}/`;
  for (const [key, url] of blobUrlCache) {
    if (key.startsWith(prefix)) {
      URL.revokeObjectURL(url);
      blobUrlCache.delete(key);
    }
  }
};

/**
 * Eagerly fetch and cache all frames for a session. Skips frames that are
 * already in IndexedDB. Reports progress via `onProgress`.
 */
export const cacheSession = async (
  sessionId: string,
  frameCount: number,
  onProgress: (cached: number, total: number) => void,
): Promise<void> => {
  let db: IDBDatabase;
  try {
    db = await openDB();
  } catch {
    // IDB unavailable — nothing to cache.
    return;
  }

  await evictIfNeeded(db);

  // Determine which frames are already cached by scanning the index.
  const existingKeys = await new Promise<Set<string>>((resolve, reject) => {
    const keys = new Set<string>();
    const tx = db.transaction(FRAMES_STORE, "readonly");
    const store = tx.objectStore(FRAMES_STORE);
    const index = store.index("sessionId");
    const cursorReq = index.openKeyCursor(IDBKeyRange.only(sessionId));

    cursorReq.onsuccess = () => {
      const cursor = cursorReq.result;
      if (cursor) {
        keys.add(cursor.primaryKey as string);
        cursor.continue();
      }
    };

    tx.oncomplete = () => resolve(keys);
    tx.onerror = () => reject(tx.error);
  });

  // Build the list of missing frame indices.
  const missing: number[] = [];
  for (let i = 0; i < frameCount; i++) {
    if (!existingKeys.has(frameKey(sessionId, i))) {
      missing.push(i);
    }
  }

  let cachedSoFar = frameCount - missing.length;
  onProgress(cachedSoFar, frameCount);

  if (missing.length === 0) {
    // Touch lastAccessedAt even if nothing was fetched.
    await updateSessionMeta(db, sessionId, frameCount, cachedSoFar, 0);
    return;
  }

  const { fetchFrameBlob } = await import("./api.ts");

  let batchBytes = 0;

  // Fetch in batches of BATCH_SIZE using allSettled so one failure doesn't
  // abort the entire session cache.
  for (let i = 0; i < missing.length; i += BATCH_SIZE) {
    const batch = missing.slice(i, i + BATCH_SIZE);

    const results = await Promise.allSettled(
      batch.map(async (frameIdx) => {
        const blob = await fetchFrameBlob(sessionId, frameIdx);
        await storeFrame(sessionId, frameIdx, blob);
        return blob.size;
      }),
    );

    for (const r of results) {
      if (r.status === "fulfilled") {
        cachedSoFar++;
        batchBytes += r.value;
      }
    }

    onProgress(cachedSoFar, frameCount);
  }

  await updateSessionMeta(db, sessionId, frameCount, cachedSoFar, batchBytes);
};
