# Canonical IR

The IR is an explicit binary wire format — **not** a `serde` dump of Rust
structs. Layout lives in `src/ir/`; field tables mirror `encode.rs` ↔
`decode.rs` exactly.

## Container

```
magic       8 bytes  "VOLEGFX1"
version     u32 LE   1
universe    u32 LE len + bytes  "vole.gfx.u1"
profile     u8       1 = Exact
body_len    u64 LE
body        exactly body_len bytes
```

All integers little-endian fixed-width (canonical by construction: a byte
string has exactly one decoding). `decode(encode(x)) == x` and, for valid
canonical byte strings, `encode(decode(b)) == b`.

## Body sections (fixed order)

1. **Objects** — `u64` count; each: kind `u8`, payload `u64` len, payload.
   - Raster: format `u8`, w `u32`, h `u32`, row-major `w·h·stride` bytes.
   - Palette: count `u32`, RGBA8 `[r,g,b,a]` per entry.
   - IndexedRaster: palette object id `u32`, w, h, `w·h` index `u32`s.
   - GeneratorField: family `u8`, param blob (semantics in Phase H).
   - Object identity = section index (content-addressing by SHA-256 of the
     canonical encoding is external to the container).
2. **Trajectories** — count; each: kind `u8` (1 = linear translation), key
   count `u32`, keys `(t u64, tx i32, ty i32)` strictly increasing in `t`.
3. **Instances** — count; each: object `u32`, order `u32`, layer `u32`,
   affine `6×i32` (a b tx c d ty), palette `u32` (`0xFFFF_FFFF` = none),
   clip flag `u8` (+ 4×i32 when set), trajectory `u32` (0 = none, else
   1-based index), visible `u8`.
4. **Events** — count; each: `t u64` (strictly increasing), op count `u32`,
   ops (tags below). Structural op order within an event is significant.

## Op tags

| Tag | Op | Payload |
|---|---|---|
| 1 | InstCreate | object, order, layer, affine, palette, clip, trajectory, visible |
| 2 | InstDelete | instance |
| 3 | InstTransform | instance, affine |
| 4 | InstTrajectory | instance, trajectory |
| 5 | InstLayer | instance, layer |
| 6 | InstVisible | instance, visible |
| 7 | InstClip | instance, clip |
| 8 | PaletteSet | object, offset, count, entries |
| 9 | CopyRegion | src rect 4×i32, dx, dy |
| 10 | MoveRegion | src rect, dx, dy |
| 11 | FillRect | rect 4×i32, color 4×u8 |
| 12 | BindResidual | algebra u8, region 4×i32, format u8, len u64, payload |

## Residual payloads (algebra-tagged)

`u64` record count, then per record: `x u32`, `y u32`, value bytes
(`Gray8`: 1; `Rgba8`: 4). Records sorted strictly ascending by `(y, x)`
(canonical). Algebra 1 = sparse overwrite, 2 = XOR. Coordinates are absolute
canonical-surface pixels.

## Hostile input

- Decoder checks every count against the U1 caps **before** allocation.
- Payload lengths are cross-checked against declared dimensions.
- Unknown tags/versions/universes/profile fail closed.
- No parser panics: all reads bounds-checked (`Reject::Truncated` etc.).
- Semantic validation (references, orderings, live-id uniqueness, overflow
  guards) is a second pass (`ir::validate`).
- Canonical-form verification is `ir::canonical::check_canonical`.
