"use client";

/**
 * File input with drag-and-drop and a local preview.
 *
 * Client-side validation mirrors the server's guards rather than replacing
 * them: the API re-checks type, size and decoded pixel count, because anything
 * enforced only in the browser is not enforced. Checking here as well just
 * turns a 413 round trip into an immediate message.
 */

import { useCallback, useEffect, useRef, useState } from "react";

const ACCEPTED = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"];
const MAX_BYTES = 32 * 1024 * 1024;

export function Dropzone({
  label,
  file,
  onFile,
}: {
  label: string;
  file: File | null;
  onFile: (file: File | null) => void;
}) {
  const [over, setOver] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const input = useRef<HTMLInputElement>(null);

  // Object URLs are revoked on change and unmount; leaking one per selection
  // holds the whole decoded bitmap alive for the life of the page.
  useEffect(() => {
    if (!file) {
      setPreview(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const accept = useCallback(
    (candidate: File | undefined) => {
      setError(null);
      if (!candidate) return;

      const suffix = candidate.name.slice(candidate.name.lastIndexOf(".")).toLowerCase();
      if (!ACCEPTED.includes(suffix)) {
        setError(`${suffix || "that file type"} is not a supported image`);
        return;
      }
      if (candidate.size > MAX_BYTES) {
        setError(`${(candidate.size / 1048576).toFixed(1)} MB exceeds the 32 MB limit`);
        return;
      }
      onFile(candidate);
    },
    [onFile],
  );

  return (
    <div>
      <div
        className="drop"
        data-over={over}
        data-filled={Boolean(file)}
        onClick={() => input.current?.click()}
        onDragOver={(event) => {
          event.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(event) => {
          event.preventDefault();
          setOver(false);
          accept(event.dataTransfer.files[0]);
        }}
        role="button"
        tabIndex={0}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            input.current?.click();
          }
        }}
        aria-label={`${label}: choose or drop an image`}
      >
        {preview ? (
          <>
            <img
              src={preview}
              alt=""
              style={{ display: "block", width: "100%", height: 108, objectFit: "cover" }}
            />
            <div className="filename">{file?.name}</div>
          </>
        ) : (
          <>
            <div style={{ fontSize: 12.5, fontWeight: 550 }}>{label}</div>
            <div className="hint" style={{ marginTop: 3 }}>
              Drop an image or click to browse
            </div>
          </>
        )}
      </div>

      <input
        ref={input}
        type="file"
        accept={ACCEPTED.join(",")}
        className="sr-only"
        onChange={(event) => accept(event.target.files?.[0])}
      />

      {error && (
        <p className="notice alert" style={{ marginTop: 7 }}>
          {error}
        </p>
      )}
    </div>
  );
}
