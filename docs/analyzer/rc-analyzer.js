(() => {
  let tfLoaded = false;
  let tfLoadFailed = false;
  let blazefaceModel = null;

  async function loadScript(src) {
    return new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = src;
      script.onload = resolve;
      script.onerror = reject;
      document.head.appendChild(script);
    });
  }

  async function loadTF() {
    if (tfLoaded) return true;
    if (tfLoadFailed) return false;
    try {
      if (!window.tf) {
        await loadScript('./lib/tf.min.js');
      }
      if (!window.tf) {
        tfLoadFailed = true;
        return false;
      }
      tfLoaded = true;
      return true;
    } catch (e) {
      tfLoadFailed = true;
      return false;
    }
  }

  async function loadBlazeFace() {
    if (blazefaceModel) return blazefaceModel;
    await loadTF();
    if (!window.tf) return null;
    if (!window.blazeface) {
      await loadScript('./lib/blazeface.min.js');
    }
    if (!window.blazeface) return null;
    blazefaceModel = await window.blazeface.load({
      modelUrl: './lib/blazeface-model/model.json',
    });
    return blazefaceModel;
  }

  async function detectFacesBlazeFace(canvas) {
    const model = await loadBlazeFace();
    if (!model) return [];
    try {
      const predictions = await model.estimateFaces(canvas, false);
      const faces = [];
      for (const pred of predictions) {
        const topLeft = pred.topLeft;
        const bottomRight = pred.bottomRight;
        const x = topLeft[0];
        const y = topLeft[1];
        const w = bottomRight[0] - topLeft[0];
        const h = bottomRight[1] - topLeft[1];

        const faceArea = w * h;
        const imageArea = canvas.width * canvas.height;
        if (faceArea < imageArea * 0.001) continue;

        faces.push({
          x,
          y,
          width: w,
          height: h,
          score: pred.probability[0] || pred.probability,
          landmarks: pred.landmarks || [],
        });
      }
      return faces;
    } catch (_) {
      return [];
    }
  }

  function detectFacesHeuristic(canvas, ctx) {
    const w = canvas.width;
    const h = canvas.height;
    const data = ctx.getImageData(0, 0, w, h).data;
    const skinMask = new Uint8Array(w * h);

    for (let i = 0; i < data.length; i += 4) {
      const r = data[i],
        g = data[i + 1],
        b = data[i + 2];
      if (r > 95 && g > 40 && b > 20 && r > g && r > b && r - g > 15 && r - Math.min(g, b) > 15) {
        skinMask[i / 4] = 1;
      }
    }

    const faces = [];
    const blockSize = Math.max(8, Math.floor(Math.min(w, h) / 40));
    const blocksX = Math.ceil(w / blockSize);
    const blocksY = Math.ceil(h / blockSize);
    const blockSkin = new Float32Array(blocksX * blocksY);

    for (let by = 0; by < blocksY; by++) {
      for (let bx = 0; bx < blocksX; bx++) {
        let count = 0,
          total = 0;
        for (let y = by * blockSize; y < Math.min((by + 1) * blockSize, h); y++) {
          for (let x = bx * blockSize; x < Math.min((bx + 1) * blockSize, w); x++) {
            total++;
            if (skinMask[y * w + x]) count++;
          }
        }
        blockSkin[by * blocksX + bx] = count / total;
      }
    }

    const visited = new Uint8Array(blocksX * blocksY);
    for (let by = 0; by < blocksY; by++) {
      for (let bx = 0; bx < blocksX; bx++) {
        const idx = by * blocksX + bx;
        if (visited[idx] || blockSkin[idx] < 0.4) continue;

        let minX = bx,
          maxX = bx,
          minY = by,
          maxY = by;
        const queue = [idx];
        visited[idx] = 1;
        let clusterSize = 0;

        while (queue.length > 0) {
          const ci = queue.shift();
          const cy = Math.floor(ci / blocksX);
          const cx = ci % blocksX;
          clusterSize++;
          if (cx < minX) minX = cx;
          if (cx > maxX) maxX = cx;
          if (cy < minY) minY = cy;
          if (cy > maxY) maxY = cy;

          for (const [dx, dy] of [
            [0, 1],
            [0, -1],
            [1, 0],
            [-1, 0],
          ]) {
            const nx = cx + dx,
              ny = cy + dy;
            if (nx < 0 || nx >= blocksX || ny < 0 || ny >= blocksY) continue;
            const ni = ny * blocksX + nx;
            if (!visited[ni] && blockSkin[ni] >= 0.3) {
              visited[ni] = 1;
              queue.push(ni);
            }
          }
        }

        const fw = (maxX - minX + 1) * blockSize;
        const fh = (maxY - minY + 1) * blockSize;
        const aspectRatio = fh / fw;

        if (clusterSize >= 8 && aspectRatio > 1.0 && aspectRatio < 1.8 && fw > w * 0.06 && fh > h * 0.08) {
          faces.push({
            x: minX * blockSize,
            y: minY * blockSize,
            width: fw,
            height: fh,
            score: Math.min(1, clusterSize / 20),
            landmarks: [],
          });
        }
      }
    }
    return faces;
  }

  function analyzeFrequencyDomain(canvas, ctx) {
    const size = 256;
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = size;
    tempCanvas.height = size;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(canvas, 0, 0, size, size);
    const data = tempCtx.getImageData(0, 0, size, size).data;

    const gray = new Float32Array(size * size);
    for (let i = 0; i < size * size; i++) {
      gray[i] = (data[i * 4] + data[i * 4 + 1] + data[i * 4 + 2]) / 3;
    }

    const dftSize = 64;
    const small = new Float32Array(dftSize * dftSize);
    const step = size / dftSize;
    for (let y = 0; y < dftSize; y++) {
      for (let x = 0; x < dftSize; x++) {
        small[y * dftSize + x] = gray[Math.floor(y * step) * size + Math.floor(x * step)];
      }
    }

    const spectrum = new Float32Array(dftSize * dftSize);
    const PI2 = 2 * Math.PI;

    const rowReal = new Float32Array(dftSize * dftSize);
    const rowImag = new Float32Array(dftSize * dftSize);
    for (let y = 0; y < dftSize; y++) {
      for (let k = 0; k < dftSize; k++) {
        let re = 0,
          im = 0;
        for (let x = 0; x < dftSize; x++) {
          const angle = (PI2 * k * x) / dftSize;
          re += small[y * dftSize + x] * Math.cos(angle);
          im -= small[y * dftSize + x] * Math.sin(angle);
        }
        rowReal[y * dftSize + k] = re;
        rowImag[y * dftSize + k] = im;
      }
    }

    for (let k2 = 0; k2 < dftSize; k2++) {
      for (let k1 = 0; k1 < dftSize; k1++) {
        let re = 0,
          im = 0;
        for (let y = 0; y < dftSize; y++) {
          const angle = (PI2 * k2 * y) / dftSize;
          const rr = rowReal[y * dftSize + k1];
          const ri = rowImag[y * dftSize + k1];
          re += rr * Math.cos(angle) - ri * Math.sin(angle);
          im += rr * -Math.sin(angle) + ri * Math.cos(angle);
        }
        spectrum[k2 * dftSize + k1] = Math.sqrt(re * re + im * im);
      }
    }

    const half = dftSize / 2;
    const radialBins = half;
    const radialPower = new Float32Array(radialBins);
    const radialCount = new Float32Array(radialBins);

    for (let y = 0; y < dftSize; y++) {
      for (let x = 0; x < dftSize; x++) {
        const dy = y < half ? y : y - dftSize;
        const dx = x < half ? x : x - dftSize;
        const r = Math.sqrt(dx * dx + dy * dy);
        const bin = Math.min(Math.floor(r), radialBins - 1);
        radialPower[bin] += spectrum[y * dftSize + x];
        radialCount[bin]++;
      }
    }

    for (let i = 0; i < radialBins; i++) {
      if (radialCount[i] > 0) radialPower[i] /= radialCount[i];
    }

    const lowFreqPower = radialPower.slice(1, 8).reduce((a, b) => a + b, 0);
    const midFreqPower = radialPower.slice(8, 20).reduce((a, b) => a + b, 0);
    const highFreqPower = radialPower.slice(20).reduce((a, b) => a + b, 0);

    const totalPower = lowFreqPower + midFreqPower + highFreqPower;
    const highFreqRatio = highFreqPower / (totalPower || 1);
    const midFreqRatio = midFreqPower / (totalPower || 1);

    const freqScore =
      highFreqRatio < 0.03
        ? 90
        : highFreqRatio < 0.08
        ? 75
        : highFreqRatio < 0.15
        ? 55
        : highFreqRatio < 0.25
        ? 35
        : 15;

    const vizCanvas = document.createElement('canvas');
    vizCanvas.width = dftSize;
    vizCanvas.height = dftSize;
    const vizCtx = vizCanvas.getContext('2d');
    const vizData = vizCtx.createImageData(dftSize, dftSize);

    const maxSpec = Math.max(...spectrum) || 1;
    for (let i = 0; i < dftSize * dftSize; i++) {
      const y = (i / dftSize) | 0;
      const x = i % dftSize;
      const sy = (y + half) % dftSize;
      const sx = (x + half) % dftSize;
      const val = Math.log(1 + spectrum[sy * dftSize + sx]) / Math.log(1 + maxSpec);
      const t = val;
      let sr, sg, sb;
      if (t < 0.25) {
        sr = 0;
        sg = Math.round(t * 4 * 255);
        sb = 255;
      } else if (t < 0.5) {
        sr = 0;
        sg = 255;
        sb = Math.round((1 - (t - 0.25) * 4) * 255);
      } else if (t < 0.75) {
        sr = Math.round((t - 0.5) * 4 * 255);
        sg = 255;
        sb = 0;
      } else {
        sr = 255;
        sg = Math.round((1 - (t - 0.75) * 4) * 255);
        sb = 0;
      }
      vizData.data[i * 4] = sr;
      vizData.data[i * 4 + 1] = sg;
      vizData.data[i * 4 + 2] = sb;
      vizData.data[i * 4 + 3] = 255;
    }
    vizCtx.putImageData(vizData, 0, 0);

    return {
      freqScore,
      highFreqRatio,
      midFreqRatio,
      radialPower,
      spectrumCanvas: vizCanvas,
    };
  }

  function analyzeNoisePattern(canvas, ctx) {
    const w = canvas.width,
      h = canvas.height;
    const data = ctx.getImageData(0, 0, w, h).data;

    const blockSize = 16;
    const blocksX = Math.ceil(w / blockSize);
    const blocksY = Math.ceil(h / blockSize);
    const blockNoiseStd = [];

    for (let by = 0; by < blocksY; by++) {
      for (let bx = 0; bx < blocksX; bx++) {
        const noiseVals = [];
        for (let y = by * blockSize + 1; y < Math.min((by + 1) * blockSize, h) - 1; y++) {
          for (let x = bx * blockSize + 1; x < Math.min((bx + 1) * blockSize, w) - 1; x++) {
            const idx = (y * w + x) * 4;
            for (let c = 0; c < 3; c++) {
              const center = data[idx + c];
              const avg =
                (data[idx - 4 + c] +
                  data[idx + 4 + c] +
                  data[(y - 1) * w * 4 + x * 4 + c] +
                  data[(y + 1) * w * 4 + x * 4 + c]) /
                4;
              noiseVals.push(center - avg);
            }
          }
        }

        if (noiseVals.length > 0) {
          const mean = noiseVals.reduce((a, b) => a + b, 0) / noiseVals.length;
          const std = Math.sqrt(noiseVals.reduce((a, b) => a + (b - mean) ** 2, 0) / noiseVals.length);
          blockNoiseStd.push(std);
        }
      }
    }

    const meanStd = blockNoiseStd.reduce((a, b) => a + b, 0) / blockNoiseStd.length;
    const stdOfStd = Math.sqrt(blockNoiseStd.reduce((a, b) => a + (b - meanStd) ** 2, 0) / blockNoiseStd.length);
    const coeffOfVariation = stdOfStd / (meanStd || 1);

    const noiseScore =
      coeffOfVariation < 0.12
        ? 90
        : coeffOfVariation < 0.2
        ? 70
        : coeffOfVariation < 0.35
        ? 50
        : coeffOfVariation < 0.5
        ? 30
        : 10;

    return { noiseScore, coeffOfVariation, meanNoiseStd: meanStd };
  }

  function analyzeColorHistogram(canvas, ctx) {
    const w = canvas.width,
      h = canvas.height;
    const data = ctx.getImageData(0, 0, w, h).data;

    const histR = new Float32Array(256);
    const histG = new Float32Array(256);
    const histB = new Float32Array(256);
    const pixels = w * h;

    for (let i = 0; i < data.length; i += 4) {
      histR[data[i]]++;
      histG[data[i + 1]]++;
      histB[data[i + 2]]++;
    }

    for (let i = 0; i < 256; i++) {
      histR[i] /= pixels;
      histG[i] /= pixels;
      histB[i] /= pixels;
    }

    function histSmoothness(hist) {
      let totalDiff = 0;
      for (let i = 1; i < 255; i++) {
        totalDiff += Math.abs(hist[i + 1] - 2 * hist[i] + hist[i - 1]);
      }
      return totalDiff;
    }

    const smoothR = histSmoothness(histR);
    const smoothG = histSmoothness(histG);
    const smoothB = histSmoothness(histB);
    const avgSmooth = (smoothR + smoothG + smoothB) / 3;

    const colorScore =
      avgSmooth < 0.0003
        ? 85
        : avgSmooth < 0.0008
        ? 70
        : avgSmooth < 0.002
        ? 50
        : avgSmooth < 0.004
        ? 30
        : 10;

    const satValues = [];
    for (let i = 0; i < data.length; i += 16) {
      const r = data[i],
        g = data[i + 1],
        b = data[i + 2];
      const max = Math.max(r, g, b);
      const min = Math.min(r, g, b);
      if (max > 0) satValues.push((max - min) / max);
    }
    const meanSat = satValues.reduce((a, b) => a + b, 0) / satValues.length;
    const satStd = Math.sqrt(satValues.reduce((a, b) => a + (b - meanSat) ** 2, 0) / satValues.length);

    return { colorScore, avgSmooth, satMean: meanSat, satStd };
  }

  function analyzeTextureRegularity(canvas, ctx) {
    const w = canvas.width,
      h = canvas.height;
    const data = ctx.getImageData(0, 0, w, h).data;

    const lbpHist = new Float32Array(256);
    let count = 0;
    const step = Math.max(1, Math.floor(Math.min(w, h) / 256));

    for (let y = 1; y < h - 1; y += step) {
      for (let x = 1; x < w - 1; x += step) {
        const idx = y * w + x;
        const center = (data[idx * 4] + data[idx * 4 + 1] + data[idx * 4 + 2]) / 3;

        let lbp = 0;
        const neighbors = [
          (y - 1) * w + (x - 1),
          (y - 1) * w + x,
          (y - 1) * w + (x + 1),
          y * w + (x + 1),
          (y + 1) * w + (x + 1),
          (y + 1) * w + x,
          (y + 1) * w + (x - 1),
          y * w + (x - 1),
        ];

        for (let n = 0; n < 8; n++) {
          const ni = neighbors[n];
          const nVal = (data[ni * 4] + data[ni * 4 + 1] + data[ni * 4 + 2]) / 3;
          if (nVal >= center) lbp |= 1 << n;
        }

        lbpHist[lbp]++;
        count++;
      }
    }

    for (let i = 0; i < 256; i++) lbpHist[i] /= count;

    let entropy = 0;
    for (let i = 0; i < 256; i++) {
      if (lbpHist[i] > 0) entropy -= lbpHist[i] * Math.log2(lbpHist[i]);
    }

    const textureScore =
      entropy < 4.5
        ? 85
        : entropy < 5.5
        ? 70
        : entropy < 6.2
        ? 50
        : entropy < 7
        ? 30
        : 12;

    return { textureScore, lbpEntropy: entropy };
  }

  function analyzeSymmetry(canvas, ctx, faces) {
    if (!faces || faces.length === 0) return { symmetryScore: 30, symmetryValue: 0.5 };

    const w = canvas.width,
      h = canvas.height;
    const data = ctx.getImageData(0, 0, w, h).data;

    let totalSymmetry = 0;
    let faceCount = 0;

    for (const face of faces) {
      const fx = Math.max(0, Math.floor(face.x));
      const fy = Math.max(0, Math.floor(face.y));
      const fw = Math.min(Math.floor(face.width), w - fx);
      const fh = Math.min(Math.floor(face.height), h - fy);
      if (fw < 10 || fh < 10) continue;

      const halfW = Math.floor(fw / 2);
      let diffSum = 0,
        count = 0;

      for (let y = fy; y < fy + fh; y++) {
        for (let x = 0; x < halfW; x++) {
          const leftIdx = (y * w + (fx + x)) * 4;
          const rightIdx = (y * w + (fx + fw - 1 - x)) * 4;
          if (leftIdx >= 0 && rightIdx >= 0 && leftIdx < data.length && rightIdx < data.length) {
            for (let c = 0; c < 3; c++) {
              diffSum += Math.abs(data[leftIdx + c] - data[rightIdx + c]);
            }
            count++;
          }
        }
      }

      const avgDiff = count > 0 ? diffSum / (count * 3) : 50;
      totalSymmetry += avgDiff;
      faceCount++;
    }

    if (faceCount === 0) return { symmetryScore: 30, symmetryValue: 0.5 };

    const avgSymmetry = totalSymmetry / faceCount;
    const symmetryScore =
      avgSymmetry < 6
        ? 85
        : avgSymmetry < 10
        ? 70
        : avgSymmetry < 16
        ? 50
        : avgSymmetry < 25
        ? 30
        : 12;

    return { symmetryScore, symmetryValue: avgSymmetry };
  }

  function detectScreenshot(canvas, ctx) {
    const w = canvas.width,
      h = canvas.height;
    const data = ctx.getImageData(0, 0, w, h).data;
    const totalPixels = w * h;

    const blockSize = 16;
    const blocksX = Math.ceil(w / blockSize);
    const blocksY = Math.ceil(h / blockSize);
    let flatBlocks = 0,
      totalBlocks = 0;

    for (let by = 0; by < blocksY; by++) {
      for (let bx = 0; bx < blocksX; bx++) {
        let minR = 255,
          maxR = 0,
          minG = 255,
          maxG = 0,
          minB = 255,
          maxB = 0;
        for (let y = by * blockSize; y < Math.min((by + 1) * blockSize, h); y++) {
          for (let x = bx * blockSize; x < Math.min((bx + 1) * blockSize, w); x++) {
            const i = (y * w + x) * 4;
            const r = data[i],
              g = data[i + 1],
              b = data[i + 2];
            if (r < minR) minR = r;
            if (r > maxR) maxR = r;
            if (g < minG) minG = g;
            if (g > maxG) maxG = g;
            if (b < minB) minB = b;
            if (b > maxB) maxB = b;
          }
        }
        totalBlocks++;
        if (maxR - minR < 5 && maxG - minG < 5 && maxB - minB < 5) flatBlocks++;
      }
    }
    const flatRatio = flatBlocks / totalBlocks;

    const colorSet = new Set();
    const sampleStep = Math.max(1, Math.floor(totalPixels / 10000));
    for (let i = 0; i < data.length; i += sampleStep * 4) {
      const r = data[i] >> 2,
        g = data[i + 1] >> 2,
        b = data[i + 2] >> 2;
      colorSet.add((r << 16) | (g << 8) | b);
    }
    const uniqueColors = colorSet.size;
    const lowColorDiversity = uniqueColors < 500;

    let noiseSum = 0,
      noiseCount = 0;
    for (let y = 1; y < h - 1; y += 3) {
      for (let x = 1; x < w - 1; x += 3) {
        const idx = (y * w + x) * 4;
        for (let c = 0; c < 3; c++) {
          const center = data[idx + c];
          const avg =
            (data[idx - 4 + c] +
              data[idx + 4 + c] +
              data[(y - 1) * w * 4 + x * 4 + c] +
              data[(y + 1) * w * 4 + x * 4 + c]) /
            4;
          noiseSum += Math.abs(center - avg);
          noiseCount++;
        }
      }
    }
    const avgNoise = noiseSum / noiseCount;
    const veryLowNoise = avgNoise < 1.5;

    let indicators = 0;
    if (flatRatio > 0.35) indicators++;
    if (lowColorDiversity) indicators++;
    if (veryLowNoise) indicators++;
    if (flatRatio > 0.5) indicators++;

    const isScreenshot = indicators >= 2;
    return { isScreenshot, flatRatio, uniqueColors, avgNoise, indicators };
  }

  function analyzeEdgeCoherence(canvas, ctx) {
    const w = canvas.width,
      h = canvas.height;
    const data = ctx.getImageData(0, 0, w, h).data;

    const edgeStrengths = [];
    const edgeDirections = [];

    for (let y = 1; y < h - 1; y += 2) {
      for (let x = 1; x < w - 1; x += 2) {
        const getG = (px, py) => {
          const i = (py * w + px) * 4;
          return (data[i] + data[i + 1] + data[i + 2]) / 3;
        };

        const gx =
          -getG(x - 1, y - 1) -
          2 * getG(x - 1, y) -
          getG(x - 1, y + 1) +
          getG(x + 1, y - 1) +
          2 * getG(x + 1, y) +
          getG(x + 1, y + 1);
        const gy =
          -getG(x - 1, y - 1) -
          2 * getG(x, y - 1) -
          getG(x + 1, y - 1) +
          getG(x - 1, y + 1) +
          2 * getG(x, y + 1) +
          getG(x + 1, y + 1);

        const mag = Math.sqrt(gx * gx + gy * gy);
        if (mag > 20) {
          edgeStrengths.push(mag);
          edgeDirections.push(Math.atan2(gy, gx));
        }
      }
    }

    if (edgeStrengths.length < 10) return { edgeScore: 30, edgeCoherence: 0.5 };

    const dirBins = 36;
    const dirHist = new Float32Array(dirBins);
    for (const dir of edgeDirections) {
      const bin = Math.floor(((dir + Math.PI) / (2 * Math.PI)) * dirBins) % dirBins;
      dirHist[bin]++;
    }

    const totalEdges = edgeStrengths.length;
    for (let i = 0; i < dirBins; i++) dirHist[i] /= totalEdges;

    let dirEntropy = 0;
    for (let i = 0; i < dirBins; i++) {
      if (dirHist[i] > 0) dirEntropy -= dirHist[i] * Math.log2(dirHist[i]);
    }

    const meanEdge = edgeStrengths.reduce((a, b) => a + b, 0) / totalEdges;
    const edgeStd = Math.sqrt(edgeStrengths.reduce((a, b) => a + (b - meanEdge) ** 2, 0) / totalEdges);
    const edgeCV = edgeStd / (meanEdge || 1);

    const edgeScore =
      edgeCV < 0.6
        ? 80
        : edgeCV < 0.9
        ? 60
        : edgeCV < 1.2
        ? 40
        : edgeCV < 1.5
        ? 20
        : 10;

    return { edgeScore, edgeCoherence: edgeCV, dirEntropy };
  }

  function analyzeSkinWithFaces(canvas, ctx, faces) {
    const w = canvas.width,
      h = canvas.height;
    const data = ctx.getImageData(0, 0, w, h).data;
    const skinMask = new Uint8Array(w * h);
    let skinCount = 0;

    for (let i = 0; i < data.length; i += 4) {
      const r = data[i],
        g = data[i + 1],
        b = data[i + 2];
      if (r > 95 && g > 40 && b > 20 && r > g && r > b && r - g > 15 && r - Math.min(g, b) > 15) {
        skinMask[i / 4] = 1;
        skinCount++;
      }
    }

    const totalPixels = w * h;
    const skinRatio = skinCount / totalPixels;

    if (skinRatio < 0.02) {
      return { hasSkin: false, skinRatio, avgSmoothness: 0, isSmoothed: false, visualization: null, faceSmoothing: [] };
    }

    let smoothnessSum = 0,
      smoothnessCount = 0;
    const faceSmoothing = [];

    if (faces && faces.length > 0) {
      for (const face of faces) {
        const fx = Math.max(1, Math.floor(face.x));
        const fy = Math.max(1, Math.floor(face.y));
        const fw = Math.min(Math.floor(face.width), w - fx - 1);
        const fh = Math.min(Math.floor(face.height), h - fy - 1);

        const regions = [
          { name: '额头', y1: fy, y2: fy + Math.floor(fh * 0.3), x1: fx + Math.floor(fw * 0.2), x2: fx + Math.floor(fw * 0.8) },
          { name: '左脸颊', y1: fy + Math.floor(fh * 0.3), y2: fy + Math.floor(fh * 0.7), x1: fx, x2: fx + Math.floor(fw * 0.35) },
          { name: '右脸颊', y1: fy + Math.floor(fh * 0.3), y2: fy + Math.floor(fh * 0.7), x1: fx + Math.floor(fw * 0.65), x2: fx + fw },
          { name: '下巴', y1: fy + Math.floor(fh * 0.75), y2: fy + fh, x1: fx + Math.floor(fw * 0.2), x2: fx + Math.floor(fw * 0.8) },
        ];

        const regionResults = [];
        for (const region of regions) {
          let lapSum = 0,
            lapCount = 0;
          for (let y = Math.max(1, region.y1); y < Math.min(h - 1, region.y2); y++) {
            for (let x = Math.max(1, region.x1); x < Math.min(w - 1, region.x2); x++) {
              if (!skinMask[y * w + x]) continue;
              const getGray = (px) => (data[px * 4] + data[px * 4 + 1] + data[px * 4 + 2]) / 3;
              const idx = y * w + x;
              const lap = Math.abs(
                getGray(idx - 1) + getGray(idx + 1) + getGray(idx - w) + getGray(idx + w) - 4 * getGray(idx)
              );
              lapSum += lap;
              lapCount++;
            }
          }
          const avgLap = lapCount > 0 ? lapSum / lapCount : 10;
          regionResults.push({ name: region.name, smoothness: avgLap, isSmoothed: avgLap < 6, pixelCount: lapCount });
        }

        faceSmoothing.push({ regions: regionResults });
      }
    }

    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const idx = y * w + x;
        if (!skinMask[idx]) continue;
        const getGray = (px) => (data[px * 4] + data[px * 4 + 1] + data[px * 4 + 2]) / 3;
        const lap = Math.abs(getGray(idx - 1) + getGray(idx + 1) + getGray(idx - w) + getGray(idx + w) - 4 * getGray(idx));
        smoothnessSum += lap;
        smoothnessCount++;
      }
    }

    const avgSmoothness = smoothnessCount > 0 ? smoothnessSum / smoothnessCount : 0;
    return {
      hasSkin: true,
      skinRatio,
      avgSmoothness,
      isSmoothed: avgSmoothness < 6,
      visualization: null,
      faceSmoothing,
    };
  }

  function performELA(canvas, ctx, img) {
    const w = img.width,
      h = img.height;
    const originalData = ctx.getImageData(0, 0, w, h);

    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = w;
    tempCanvas.height = h;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(img, 0, 0);
    const jpegDataUrl = tempCanvas.toDataURL('image/jpeg', 0.95);

    const recompImg = new Image();
    return new Promise((resolve) => {
      recompImg.onload = () => {
        tempCtx.drawImage(recompImg, 0, 0);
        const recompData = tempCtx.getImageData(0, 0, w, h);
        const diffData = ctx.createImageData(w, h);
        let totalDiff = 0;
        const blockSize = 16;
        const blocksX = Math.ceil(w / blockSize);
        const blocksY = Math.ceil(h / blockSize);
        const blockAvgs = [];

        for (let i = 0; i < originalData.data.length; i += 4) {
          const dr = Math.abs(originalData.data[i] - recompData.data[i]);
          const dg = Math.abs(originalData.data[i + 1] - recompData.data[i + 1]);
          const db = Math.abs(originalData.data[i + 2] - recompData.data[i + 2]);
          const diff = (dr + dg + db) / 3;
          totalDiff += diff;

          const scaled = Math.min(255, diff * 10);
          const t = scaled / 255;
          let r, g, b;
          if (t < 0.25) {
            r = 0;
            g = Math.round(t * 4 * 255);
            b = 255;
          } else if (t < 0.5) {
            r = 0;
            g = 255;
            b = Math.round((1 - (t - 0.25) * 4) * 255);
          } else if (t < 0.75) {
            r = Math.round((t - 0.5) * 4 * 255);
            g = 255;
            b = 0;
          } else {
            r = 255;
            g = Math.round((1 - (t - 0.75) * 4) * 255);
            b = 0;
          }
          diffData.data[i] = r;
          diffData.data[i + 1] = g;
          diffData.data[i + 2] = b;
          diffData.data[i + 3] = 255;
        }

        const pixelCount = w * h;
        const avgDiff = totalDiff / pixelCount;

        for (let by = 0; by < blocksY; by++) {
          for (let bx = 0; bx < blocksX; bx++) {
            let sum = 0,
              count = 0;
            for (let y = by * blockSize; y < Math.min((by + 1) * blockSize, h); y++) {
              for (let x = bx * blockSize; x < Math.min((bx + 1) * blockSize, w); x++) {
                const idx = (y * w + x) * 4;
                sum +=
                  (Math.abs(originalData.data[idx] - recompData.data[idx]) +
                    Math.abs(originalData.data[idx + 1] - recompData.data[idx + 1]) +
                    Math.abs(originalData.data[idx + 2] - recompData.data[idx + 2])) /
                  3;
                count++;
              }
            }
            blockAvgs.push(sum / count);
          }
        }

        const blockMean = blockAvgs.reduce((a, b) => a + b, 0) / blockAvgs.length;
        const blockVariance = blockAvgs.reduce((a, b) => a + (b - blockMean) ** 2, 0) / blockAvgs.length;

        resolve({ heatmap: diffData, avgDiff, blockVariance, blockMean });
      };
      recompImg.src = jpegDataUrl;
    });
  }

  function calculateScore(elaResult, skinResult, aiProbability) {
    const elaScore = Math.max(0, Math.min(100, 100 - elaResult.blockVariance * 2));
    const textureScore = Math.max(
      0,
      Math.min(100, elaResult.avgDiff < 1 ? 30 : elaResult.avgDiff < 5 ? 80 : elaResult.avgDiff < 15 ? 60 : 40)
    );
    const noiseScore = Math.max(0, Math.min(100, 100 - Math.sqrt(elaResult.blockVariance) * 5));

    let skinScore = 70;
    if (skinResult.hasSkin) {
      skinScore = Math.max(
        0,
        Math.min(
          100,
          skinResult.avgSmoothness < 2
            ? 10
            : skinResult.avgSmoothness < 4
            ? 25
            : skinResult.avgSmoothness < 6
            ? 40
            : skinResult.avgSmoothness < 8
            ? 60
            : skinResult.avgSmoothness < 12
            ? 75
            : 90
        )
      );
    }

    const aiRealityScore = 100 - aiProbability;

    let overall;
    if (skinResult.hasSkin) {
      const weighted = Math.round(elaScore * 0.15 + textureScore * 0.05 + noiseScore * 0.05 + skinScore * 0.3 + aiRealityScore * 0.45);
      const cap = aiProbability > 50 || skinScore < 30 ? 45 : aiProbability > 35 || skinScore < 45 ? 60 : 100;
      overall = Math.min(weighted, cap);
    } else {
      const weighted = Math.round(elaScore * 0.2 + textureScore * 0.05 + noiseScore * 0.05 + aiRealityScore * 0.7);
      const cap = aiProbability > 50 ? 40 : aiProbability > 35 ? 55 : 100;
      overall = Math.min(weighted, cap);
    }

    return {
      overall: Math.max(0, Math.min(100, overall)),
      elaScore: Math.round(elaScore),
      textureScore: Math.round(textureScore),
      noiseScore: Math.round(noiseScore),
      skinScore: Math.round(skinScore),
    };
  }

  function loadImageFromDataUrl(dataUrl) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error('Failed to load image data'));
      img.src = dataUrl;
    });
  }

  async function analyzePayload(payload) {
    const mime = (payload && payload.meta && payload.meta.mime) || 'image/jpeg';
    const includeImages = payload && payload.includeImages;
    const dataUrl = `data:${mime};base64,${payload.imageBase64}`;
    const img = await loadImageFromDataUrl(dataUrl);

    const w = img.width || img.naturalWidth;
    const h = img.height || img.naturalHeight;
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);

    const screenshot = detectScreenshot(canvas, ctx);
    if (screenshot.isScreenshot) {
      return {
        summary: {
          aiGeneratedScore: 0,
          tamperScore: 0,
          beautyFilterScore: 0,
          confidence: 0.3,
          notes: ['Screenshot detected, analysis skipped.'],
        },
        screenshot,
      };
    }

    let faces = [];
    try {
      faces = await detectFacesBlazeFace(canvas);
    } catch (_) {
      faces = [];
    }
    if (faces.length === 0) {
      faces = detectFacesHeuristic(canvas, ctx);
    }

    const elaResult = await performELA(canvas, ctx, img);
    const skinResult = analyzeSkinWithFaces(canvas, ctx, faces);
    const freqResult = analyzeFrequencyDomain(canvas, ctx);
    const noiseResult = analyzeNoisePattern(canvas, ctx);
    const colorResult = analyzeColorHistogram(canvas, ctx);
    const textureResult = analyzeTextureRegularity(canvas, ctx);
    const symmetryResult = analyzeSymmetry(canvas, ctx, faces);
    const edgeResult = analyzeEdgeCoherence(canvas, ctx);

    const aiProbability = Math.round(
      freqResult.freqScore * 0.2 +
        noiseResult.noiseScore * 0.2 +
        colorResult.colorScore * 0.15 +
        textureResult.textureScore * 0.15 +
        symmetryResult.symmetryScore * 0.15 +
        edgeResult.edgeScore * 0.15
    );

    const scores = calculateScore(elaResult, skinResult, aiProbability);
    const summary = {
      aiGeneratedScore: Number((aiProbability / 100).toFixed(2)),
      tamperScore: Number((Math.max(0, 100 - scores.elaScore) / 100).toFixed(2)),
      beautyFilterScore: skinResult.hasSkin ? Number((Math.max(0, 100 - scores.skinScore) / 100).toFixed(2)) : 0,
      confidence: faces.length > 0 ? 0.75 : 0.6,
      notes: [],
    };

    const result = {
      summary,
      scores: {
        overall: scores.overall,
        aiProbability,
        elaScore: scores.elaScore,
        textureScore: scores.textureScore,
        noiseScore: scores.noiseScore,
        skinScore: scores.skinScore,
      },
      metrics: {
        ela: { avgDiff: elaResult.avgDiff, blockVariance: elaResult.blockVariance, blockMean: elaResult.blockMean },
        skin: {
          hasSkin: skinResult.hasSkin,
          skinRatio: skinResult.skinRatio,
          avgSmoothness: skinResult.avgSmoothness,
          isSmoothed: skinResult.isSmoothed,
          faceSmoothing: skinResult.faceSmoothing,
        },
        frequency: { highFreqRatio: freqResult.highFreqRatio, midFreqRatio: freqResult.midFreqRatio },
        noise: { coeffOfVariation: noiseResult.coeffOfVariation },
        color: { avgSmooth: colorResult.avgSmooth },
        texture: { lbpEntropy: textureResult.lbpEntropy },
        symmetry: { symmetryValue: symmetryResult.symmetryValue },
        edge: { edgeCoherence: edgeResult.edgeCoherence },
      },
      faces,
    };

    if (includeImages) {
      const elaCanvas = document.createElement('canvas');
      elaCanvas.width = w;
      elaCanvas.height = h;
      elaCanvas.getContext('2d').putImageData(elaResult.heatmap, 0, 0);
      result.images = {
        elaHeatmap: elaCanvas.toDataURL('image/png'),
        spectrum: freqResult.spectrumCanvas.toDataURL('image/png'),
      };
    }

    return result;
  }

  window.analyzeImage = async function (payload) {
    try {
      const result = await analyzePayload(payload);
      if (window.AnalyzerChannel && typeof window.AnalyzerChannel.postMessage === 'function') {
        window.AnalyzerChannel.postMessage(JSON.stringify(result));
      }
    } catch (err) {
      const errorPayload = { error: String(err) };
      if (window.AnalyzerChannel && typeof window.AnalyzerChannel.postMessage === 'function') {
        window.AnalyzerChannel.postMessage(JSON.stringify(errorPayload));
      }
    }
  };
})();
