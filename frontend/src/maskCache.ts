/**
 * IndexedDB + in-memory mask cache for video annotation sessions.
 *
 * Two-tier cache: an in-memory Map for instant access during a page lifetime,
 * and IndexedDB for persistence across reloads. Session-level LRU eviction
 * keeps at most MAX_SESSIONS in IDB.
 *
 * During propagation, `putMasksMemoryOnly` avoids hundreds of IDB writes.
 * Call `flushToIDB` after propagation completes to bulk-persist.
 */

import type { MaskResult } from "./types.ts";

const DB_NAME = "mask-cache";
const DB_VERSION = 1;
const MASKS_STORE = "masks";
const SESSIONS_STORE = "sessions";
const MAX_SESSIONS = 5;
const MAX_MEMORY_ENTRIES = 128;

interface MaskCacheEntry {
  masks: Record<number, MaskResult>;
  version: number;
}

interface IDBMaskRecord {
  sessionId: string;
  frameIdx: number;
  masks: Record<number, MaskResult>;
  version: number;
}

interface SessionMeta {
  sessionId: string;
  versions: Record<number, number>;
  lastAccessedAt: number;
}

// ---------------------------------------------------------------------------
// Module-level state
// ---------------------------------------------------------------------------

/** Cached IDB connection -- opened once, reused for the lifetime of the page. */
let _db: IDBDatabase | null = null;

/** In-memory mask cache: "sessionId/frameIdx" -> entry. LRU by insertion order. */
const memoryCache = new Map<string, MaskCacheEntry>();

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
      if (!db.objectStoreNames.contains(MASKS_STORE)) {
        const masksStore = db.createObjectStore(MASKS_STORE);
        masksStore.createIndex("sessionId", "sessionId", { unique: false });
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

/** Deterministic key for a single frame's masks. */
const cacheKey = (sessionId: string, frameIdx: number): string =>
  `${sessionId}/${frameIdx}`;

/**
 * Evict the oldest entry from the in-memory cache when it exceeds
 * MAX_MEMORY_ENTRIES. Map iteration order is insertion order, so the
 * first key is the least-recently-inserted.
 */
const evictMemoryIfNeeded = (): void => {
  while (memoryCache.size >= MAX_MEMORY_ENTRIES) {
    const oldest = memoryCache.keys().next().value;
    if (oldest !== undefined) {
      memoryCache.delete(oldest);
    } else {
      break;
    }
  }
};

/**
 * Write a single mask record into the IDB masks store.
 * Silently warns on first failure, does not throw.
 */
const storeMaskRecord = async (
  db: IDBDatabase,
  sessionId: string,
  frameIdx: number,
  masks: Record<number, MaskResult>,
  version: number,
): Promise<void> => {
  const record: IDBMaskRecord = { sessionId, frameIdx, masks, version };
  return new Promise<void>((resolve, reject) => {
    const tx = db.transaction(MASKS_STORE, "readwrite");
    const store = tx.objectStore(MASKS_STORE);
    store.put(record, cacheKey(sessionId, frameIdx));
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
};

/**
 * Touch the session's lastAccessedAt timestamp. Creates the session
 * metadata entry if it doesn't exist.
 */
const touchSession = async (
  db: IDBDatabase,
  sessionId: string,
): Promise<void> => {
  return new Promise<void>((resolve, reject) => {
    const tx = db.transaction(SESSIONS_STORE, "readwrite");
    const store = tx.objectStore(SESSIONS_STORE);
    const getReq = store.get(sessionId);

    getReq.onsuccess = () => {
      const existing = getReq.result as SessionMeta | undefined;
      const meta: SessionMeta = {
        sessionId,
        versions: existing?.versions ?? {},
        lastAccessedAt: Date.now(),
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
const evictIfNeeded = async (
  db: IDBDatabase,
  currentSessionId: string,
): Promise<void> => {
  const sessions = await new Promise<SessionMeta[]>((resolve, reject) => {
    const tx = db.transaction(SESSIONS_STORE, "readonly");
    const store = tx.objectStore(SESSIONS_STORE);
    const req = store.getAll();
    req.onsuccess = () => resolve(req.result as SessionMeta[]);
    req.onerror = () => reject(req.error);
  });

  if (sessions.length < MAX_SESSIONS) return;

  // Find the session with the oldest lastAccessedAt, excluding current.
  let oldest: SessionMeta | null = null;
  for (const s of sessions) {
    if (s.sessionId === currentSessionId) continue;
    if (!oldest || s.lastAccessedAt < oldest.lastAccessedAt) {
      oldest = s;
    }
  }

  if (oldest) {
    await evictSession(oldest.sessionId);
  }
};

// ---------------------------------------------------------------------------
// Exported API
// ---------------------------------------------------------------------------

/**
 * Get cached masks for a specific frame. Checks the in-memory cache first,
 * then IndexedDB. Returns null on miss -- does NOT fetch from network.
 */
export const getCachedMasks = async (
  sessionId: string,
  frameIdx: number,
): Promise<Record<number, MaskResult> | null> => {
  const key = cacheKey(sessionId, frameIdx);

  // 1. In-memory hit -- instant.
  const cached = memoryCache.get(key);
  if (cached) return cached.masks;

  // 2. Try IndexedDB.
  try {
    const db = await openDB();
    const record = await new Promise<IDBMaskRecord | undefined>(
      (resolve, reject) => {
        const tx = db.transaction(MASKS_STORE, "readonly");
        const store = tx.objectStore(MASKS_STORE);
        const req = store.get(key);
        req.onsuccess = () =>
          resolve(req.result as IDBMaskRecord | undefined);
        req.onerror = () => reject(req.error);
      },
    );

    if (record) {
      // Promote to memory cache for subsequent reads.
      evictMemoryIfNeeded();
      memoryCache.set(key, { masks: record.masks, version: record.version });
      return record.masks;
    }
  } catch {
    if (!_idbWarned) {
      _idbWarned = true;
      console.warn(
        "[maskCache] IndexedDB unavailable -- mask cache degraded to memory-only.",
      );
    }
  }

  return null;
};

/**
 * Write masks to both the in-memory cache and IndexedDB.
 */
export const putMasks = async (
  sessionId: string,
  frameIdx: number,
  masks: Record<number, MaskResult>,
  version: number = 0,
): Promise<void> => {
  const key = cacheKey(sessionId, frameIdx);

  // Write to memory.
  evictMemoryIfNeeded();
  memoryCache.set(key, { masks, version });

  // Write to IDB.
  try {
    const db = await openDB();
    await evictIfNeeded(db, sessionId);
    await storeMaskRecord(db, sessionId, frameIdx, masks, version);
    await touchSession(db, sessionId);
  } catch {
    if (!_idbWarned) {
      _idbWarned = true;
      console.warn("[maskCache] IndexedDB write failed -- memory-only mode.");
    }
  }
};

/**
 * Write masks to the in-memory cache only. Used during propagation to
 * avoid hundreds of individual IDB writes. Call `flushToIDB` after
 * propagation completes.
 */
export const putMasksMemoryOnly = (
  sessionId: string,
  frameIdx: number,
  masks: Record<number, MaskResult>,
  version: number = 0,
): void => {
  const key = cacheKey(sessionId, frameIdx);
  evictMemoryIfNeeded();
  memoryCache.set(key, { masks, version });
};

/**
 * Merge masks into a frame's cached record instead of replacing it.
 *
 * Click/box responses and propagation SSE events carry only the objects
 * involved in that result. Writing them through `putMasks` would erase the
 * other objects' masks from the cached record — which `loadSessionMasksSmart`
 * then serves as fresh on the next session load, making those objects appear
 * deleted even though the server still has them. Merging against the existing
 * record (memory first, IDB fallback) preserves them.
 */
export const mergeMasks = async (
  sessionId: string,
  frameIdx: number,
  masks: Record<number, MaskResult>,
  memoryOnly: boolean,
): Promise<void> => {
  const existing = await getCachedMasks(sessionId, frameIdx);
  const merged = existing ? { ...existing, ...masks } : masks;
  if (memoryOnly) {
    putMasksMemoryOnly(sessionId, frameIdx, merged);
  } else {
    await putMasks(sessionId, frameIdx, merged);
  }
};

/**
 * Bulk-write all in-memory entries for a session to IndexedDB.
 * Called after propagation completes to persist results.
 */
export const flushToIDB = async (sessionId: string): Promise<void> => {
  const prefix = `${sessionId}/`;
  const entries: Array<{ key: string; entry: MaskCacheEntry; frameIdx: number }> = [];

  for (const [key, entry] of memoryCache) {
    if (key.startsWith(prefix)) {
      const frameIdx = parseInt(key.slice(prefix.length), 10);
      entries.push({ key, entry, frameIdx });
    }
  }

  if (entries.length === 0) return;

  try {
    const db = await openDB();
    await evictIfNeeded(db, sessionId);

    // Batch all writes into a single transaction for performance.
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(MASKS_STORE, "readwrite");
      const store = tx.objectStore(MASKS_STORE);

      for (const { key, entry, frameIdx } of entries) {
        const record: IDBMaskRecord = {
          sessionId,
          frameIdx,
          masks: entry.masks,
          version: entry.version,
        };
        store.put(record, key);
      }

      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });

    await touchSession(db, sessionId);
  } catch {
    if (!_idbWarned) {
      _idbWarned = true;
      console.warn("[maskCache] IndexedDB flush failed.");
    }
  }
};

/**
 * Remove a frame from the in-memory cache. IDB is left untouched.
 */
export const invalidateFrame = (
  sessionId: string,
  frameIdx: number,
): void => {
  memoryCache.delete(cacheKey(sessionId, frameIdx));
};

/**
 * Clear all in-memory entries for a session. IDB persists.
 */
export const clearSessionMemory = (sessionId: string): void => {
  const prefix = `${sessionId}/`;
  for (const key of memoryCache.keys()) {
    if (key.startsWith(prefix)) {
      memoryCache.delete(key);
    }
  }
};

/**
 * Remove a session from both the in-memory cache and IndexedDB.
 */
export const evictSession = async (sessionId: string): Promise<void> => {
  // Clear memory entries.
  clearSessionMemory(sessionId);

  try {
    const db = await openDB();

    // Delete all mask records for this session via the sessionId index.
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(MASKS_STORE, "readwrite");
      const store = tx.objectStore(MASKS_STORE);
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
    // IDB unavailable -- nothing to evict.
  }
};

/**
 * Store version metadata for a session in IDB. Used to track which
 * mask versions are cached locally, enabling stale-frame detection.
 */
export const storeSessionMeta = async (
  sessionId: string,
  versions: Record<number, number>,
): Promise<void> => {
  try {
    const db = await openDB();
    return new Promise<void>((resolve, reject) => {
      const tx = db.transaction(SESSIONS_STORE, "readwrite");
      const store = tx.objectStore(SESSIONS_STORE);
      const getReq = store.get(sessionId);

      getReq.onsuccess = () => {
        const existing = getReq.result as SessionMeta | undefined;
        const meta: SessionMeta = {
          sessionId,
          versions,
          lastAccessedAt: existing?.lastAccessedAt ?? Date.now(),
        };
        store.put(meta, sessionId);
      };

      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  } catch {
    if (!_idbWarned) {
      _idbWarned = true;
      console.warn("[maskCache] Failed to store session metadata.");
    }
  }
};

/**
 * Compare local IDB versions against server versions and return frame
 * indices that are stale (local version < server version, or missing locally).
 */
export const findStaleFrames = async (
  sessionId: string,
  serverVersions: Record<number, number>,
): Promise<number[]> => {
  let localVersions: Record<number, number> = {};

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

    if (meta) {
      localVersions = meta.versions;
    }
  } catch {
    // IDB unavailable -- treat all frames as stale.
    return Object.keys(serverVersions).map(Number);
  }

  const stale: number[] = [];
  for (const frameIdxStr of Object.keys(serverVersions)) {
    const frameIdx = Number(frameIdxStr);
    const serverVersion = serverVersions[frameIdx];
    const localVersion = localVersions[frameIdx];
    if (localVersion === undefined || localVersion < serverVersion) {
      stale.push(frameIdx);
    }
  }

  return stale;
};

/**
 * Return frames that exist in local IDB for this session but are NOT in
 * the server's versions dict. These are frames that were deleted on the
 * server (e.g., from another device) and should be evicted from the cache.
 */
export const findOrphanedFrames = async (
  sessionId: string,
  serverVersions: Record<number, number>,
): Promise<number[]> => {
  const serverFrames = new Set(Object.keys(serverVersions).map(Number));

  try {
    const db = await openDB();
    const localFrames = await new Promise<number[]>((resolve, reject) => {
      const tx = db.transaction(MASKS_STORE, "readonly");
      const store = tx.objectStore(MASKS_STORE);
      const index = store.index("sessionId");
      const frames: number[] = [];
      const cursorReq = index.openCursor(IDBKeyRange.only(sessionId));
      cursorReq.onsuccess = () => {
        const cursor = cursorReq.result;
        if (cursor) {
          const rec = cursor.value as IDBMaskRecord;
          frames.push(rec.frameIdx);
          cursor.continue();
        } else {
          resolve(frames);
        }
      };
      cursorReq.onerror = () => reject(cursorReq.error);
    });

    return localFrames.filter((f) => !serverFrames.has(f));
  } catch {
    return [];
  }
};

/**
 * Remove a single frame's masks from both the in-memory cache and IDB.
 */
export const evictFrame = async (
  sessionId: string,
  frameIdx: number,
): Promise<void> => {
  const key = cacheKey(sessionId, frameIdx);
  memoryCache.delete(key);

  try {
    const db = await openDB();
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(MASKS_STORE, "readwrite");
      const store = tx.objectStore(MASKS_STORE);
      store.delete(key);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  } catch {
    // IDB unavailable — memory-only eviction is sufficient.
  }
};
