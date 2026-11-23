// Minimal PNG decoder (non-interlaced, 8 or 16-bit, grey/RGB/RGBA/grey+alpha)
// with its own zlib inflate, for environments without image APIs (Scriptable).
// decode(Uint8Array) -> { width, height, data: Uint8Array RGBA }
(function (root) {
  // ---- inflate (RFC 1950/1951), after zlib's puff.c ----
  const LBASE = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
  const LEXT = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
  const DBASE = [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577];
  const DEXT = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];
  const ORDER = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];

  function huffman(lengths, n) {
    const count = new Uint16Array(16), symbol = new Uint16Array(n), offs = new Uint16Array(16);
    for (let i = 0; i < n; i++) count[lengths[i]]++;
    for (let i = 1; i < 15; i++) offs[i + 1] = offs[i] + count[i];  // length 0 = unused symbol
    for (let i = 0; i < n; i++) if (lengths[i]) symbol[offs[lengths[i]]++] = i;
    return { count, symbol };
  }

  function inflate(src, start, expected) {
    let pos = start, bitbuf = 0, bitcnt = 0;
    let out = new Uint8Array(Math.max(expected || 0, 1024)), olen = 0;
    const need = (n) => {
      if (olen + n <= out.length) return;
      const bigger = new Uint8Array(Math.max(out.length * 2, olen + n));
      bigger.set(out.subarray(0, olen)); out = bigger;
    };
    const bits = (n) => {
      let v = bitbuf;
      while (bitcnt < n) {
        if (pos >= src.length) throw new Error("PNG data ended early");
        v |= src[pos++] << bitcnt; bitcnt += 8;
      }
      bitbuf = v >>> n; bitcnt -= n;
      return v & ((1 << n) - 1);
    };
    const decode = (h) => {
      let code = 0, first = 0, index = 0;
      for (let len = 1; len < 16; len++) {
        code |= bits(1);
        const c = h.count[len];
        if (code - c < first) return h.symbol[index + (code - first)];
        index += c; first += c; first <<= 1; code <<= 1;
      }
      throw new Error("bad Huffman code");
    };
    const codes = (lh, dh) => {
      for (;;) {
        let sym = decode(lh);
        if (sym < 256) { need(1); out[olen++] = sym; continue; }
        if (sym === 256) return;
        sym -= 257;
        const len = LBASE[sym] + bits(LEXT[sym]);
        const ds = decode(dh);
        const dist = DBASE[ds] + bits(DEXT[ds]);
        if (dist > olen) throw new Error("bad distance");
        need(len);
        for (let i = 0; i < len; i++, olen++) out[olen] = out[olen - dist];
      }
    };
    let fixedL = null, fixedD = null;
    let last = 0;
    while (!last) {
      last = bits(1);
      const type = bits(2);
      if (type === 0) {
        bitbuf = 0; bitcnt = 0;
        const len = src[pos] | (src[pos + 1] << 8); pos += 4;
        need(len); out.set(src.subarray(pos, pos + len), olen); olen += len; pos += len;
      } else if (type === 1) {
        if (!fixedL) {
          const l = new Uint8Array(288);
          for (let i = 0; i < 144; i++) l[i] = 8;
          for (let i = 144; i < 256; i++) l[i] = 9;
          for (let i = 256; i < 280; i++) l[i] = 7;
          for (let i = 280; i < 288; i++) l[i] = 8;
          fixedL = huffman(l, 288);
          fixedD = huffman(new Uint8Array(30).fill(5), 30);
        }
        codes(fixedL, fixedD);
      } else if (type === 2) {
        const nlen = bits(5) + 257, ndist = bits(5) + 1, ncode = bits(4) + 4;
        const lens = new Uint8Array(320);
        for (let i = 0; i < ncode; i++) lens[ORDER[i]] = bits(3);
        const ch = huffman(lens, 19);
        const ll = new Uint8Array(nlen + ndist);
        for (let i = 0; i < nlen + ndist;) {
          const sym = decode(ch);
          if (sym < 16) { ll[i++] = sym; continue; }
          let rep, val = 0;
          if (sym === 16) { val = ll[i - 1]; rep = 3 + bits(2); }
          else if (sym === 17) rep = 3 + bits(3);
          else rep = 11 + bits(7);
          while (rep--) ll[i++] = val;
        }
        codes(huffman(ll.subarray(0, nlen), nlen), huffman(ll.subarray(nlen), ndist));
      } else throw new Error("bad block type");
    }
    return out.subarray(0, olen);
  }

  // ---- PNG ----
  function decode(bytes) {
    const sig = [137, 80, 78, 71, 13, 10, 26, 10];
    for (let i = 0; i < 8; i++) if (bytes[i] !== sig[i]) throw new Error("not a PNG");
    const u32 = (o) => ((bytes[o] << 24) | (bytes[o + 1] << 16) | (bytes[o + 2] << 8) | bytes[o + 3]) >>> 0;
    let pos = 8, width = 0, height = 0, depth = 0, ctype = 0, interlace = 0;
    const idat = [];
    let idatLen = 0;
    while (pos < bytes.length) {
      const len = u32(pos), type = String.fromCharCode(bytes[pos + 4], bytes[pos + 5], bytes[pos + 6], bytes[pos + 7]);
      const d = pos + 8;
      if (type === "IHDR") { width = u32(d); height = u32(d + 4); depth = bytes[d + 8]; ctype = bytes[d + 9]; interlace = bytes[d + 12]; }
      else if (type === "IDAT") { idat.push(bytes.subarray(d, d + len)); idatLen += len; }
      else if (type === "IEND") break;
      pos = d + len + 4;
    }
    if (interlace) throw new Error("interlaced PNG not supported");
    const channels = { 0: 1, 2: 3, 4: 2, 6: 4 }[ctype];
    if (!channels || (depth !== 8 && depth !== 16)) throw new Error(`unsupported PNG (colour type ${ctype}, ${depth}-bit)`);
    const z = new Uint8Array(idatLen);
    let o = 0;
    for (const part of idat) { z.set(part, o); o += part.length; }
    const bpp = channels * depth / 8, stride = width * bpp;
    const raw = inflate(z, 2, height * (stride + 1));
    const img = new Uint8Array(width * height * 4), prev = new Uint8Array(stride), line = new Uint8Array(stride);
    let p = 0;
    for (let y = 0; y < height; y++) {
      const f = raw[p++];
      for (let i = 0; i < stride; i++) {
        const x = raw[p++], a = i >= bpp ? line[i - bpp] : 0, b = prev[i], c = i >= bpp ? prev[i - bpp] : 0;
        let v;
        if (f === 0) v = x;
        else if (f === 1) v = x + a;
        else if (f === 2) v = x + b;
        else if (f === 3) v = x + ((a + b) >> 1);
        else {
          const pp = a + b - c, pa = Math.abs(pp - a), pb = Math.abs(pp - b), pc = Math.abs(pp - c);
          v = x + (pa <= pb && pa <= pc ? a : pb <= pc ? b : c);
        }
        line[i] = v & 255;
      }
      const step = depth / 8;  // for 16-bit keep the high byte
      for (let xI = 0; xI < width; xI++) {
        const s = xI * bpp, t = (y * width + xI) * 4;
        const ch = (k) => line[s + k * step];
        if (channels >= 3) { img[t] = ch(0); img[t + 1] = ch(1); img[t + 2] = ch(2); img[t + 3] = channels === 4 ? ch(3) : 255; }
        else { img[t] = img[t + 1] = img[t + 2] = ch(0); img[t + 3] = channels === 2 ? ch(1) : 255; }
      }
      prev.set(line);
    }
    return { width, height, data: img };
  }

  const api = { decode, inflate };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.MergePNG = api;
})(this);
