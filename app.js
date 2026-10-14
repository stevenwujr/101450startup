let scatterChartInstance;
let activeLoadRequest = 0;

function rowsToTable(rows, maxRows = 8) {
  const columns = Object.keys(rows[0] || {});
  const limited = rows.slice(0, maxRows);

  let html = '<table><thead><tr>';
  columns.forEach((col) => {
    html += `<th>${window.DatasetUtils.escapeHTML(col)}</th>`;
  });
  html += '</tr></thead><tbody>';

  limited.forEach((row) => {
    html += '<tr>';
    columns.forEach((col) => {
      const value = row[col];
      const safeValue = typeof value === 'number'
        ? value.toLocaleString(undefined, { maximumFractionDigits: 2 })
        : window.DatasetUtils.escapeHTML(value);
      html += `<td>${safeValue}</td>`;
    });
    html += '</tr>';
  });
  html += '</tbody></table>';

  return html;
}

function splitRows(length) {
  const indices = Array.from({ length }, (_, index) => index);
  let value = 42;
  const random = () => {
    value += 0x6d2b79f5;
    let mixed = value;
    mixed = Math.imul(mixed ^ (mixed >>> 15), mixed | 1);
    mixed ^= mixed + Math.imul(mixed ^ (mixed >>> 7), mixed | 61);
    return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296;
  };
  for (let index = indices.length - 1; index > 0; index--) {
    const swapIndex = Math.floor(random() * (index + 1));
    [indices[index], indices[swapIndex]] = [indices[swapIndex], indices[index]];
  }
  const testCount = Math.ceil(length * 0.2);
  return { train: indices.slice(testCount), test: indices.slice(0, testCount) };
}

function transpose(matrix) {
  return matrix[0].map((_, colIndex) => matrix.map((row) => row[colIndex]));
}

function multiplyMatrix(A, B) {
  return A.map((rowA) =>
    B[0].map((_, colIndex) =>
      rowA.reduce((sum, value, rowIndex) => sum + value * B[rowIndex][colIndex], 0)
    )
  );
}

function solveLinearSystem(A, b) {
  const n = A.length;
  const aug = A.map((row, i) => [...row, b[i]]);

  for (let i = 0; i < n; i++) {
    let pivot = i;
    for (let j = i + 1; j < n; j++) {
      if (Math.abs(aug[j][i]) > Math.abs(aug[pivot][i])) pivot = j;
    }

    if (Math.abs(aug[pivot][i]) < 1e-12) {
      throw new Error('The regression matrix is singular.');
    }

    if (pivot !== i) {
      [aug[i], aug[pivot]] = [aug[pivot], aug[i]];
    }

    const pivotValue = aug[i][i];
    for (let j = i; j <= n; j++) aug[i][j] /= pivotValue;

    for (let r = 0; r < n; r++) {
      if (r === i) continue;
      const factor = aug[r][i];
      if (Math.abs(factor) < 1e-12) continue;
      for (let j = i; j <= n; j++) {
        aug[r][j] -= factor * aug[i][j];
      }
    }
  }

  return aug.map((row) => row[n]);
}

function computeMetrics(yTrue, yPred) {
  const n = yTrue.length;
  const mae = yTrue.reduce((sum, value, i) => sum + Math.abs(value - yPred[i]), 0) / n;
  const mse = yTrue.reduce((sum, value, i) => sum + (value - yPred[i]) ** 2, 0) / n;
  const rmse = Math.sqrt(mse);
  const meanY = yTrue.reduce((sum, value) => sum + value, 0) / n;
  const sse = yTrue.reduce((sum, value, i) => sum + (value - yPred[i]) ** 2, 0);
  const sst = yTrue.reduce((sum, value) => sum + (value - meanY) ** 2, 0);
  const r2 = 1 - sse / sst;

  return { mae, mse, rmse, r2 };
}

function fitLinearRegression(rows, features, target) {
  const split = splitRows(rows.length);
  const X = split.train.map((index) => [1, ...features.map((feature) => rows[index][feature.key])]);
  const y = split.train.map((index) => rows[index][target]);
  const Xt = transpose(X);
  const XtX = multiplyMatrix(Xt, X);
  const Xty = multiplyMatrix(Xt, y.map((value) => [value]));
  const b = Xty.map((row) => row[0]);
  const coefficients = solveLinearSystem(XtX, b);

  const yTrue = split.test.map((index) => rows[index][target]);
  const yPred = split.test.map((index) => [1, ...features.map((feature) => rows[index][feature.key])]
    .reduce((sum, value, featureIndex) => sum + value * coefficients[featureIndex], 0));

  const metrics = computeMetrics(yTrue, yPred);
  return {
    coefficients,
    featureNames: ['Intercept', ...features.map((feature) => feature.label)],
    metrics,
    yPred,
    yTrue,
    split,
  };
}

function renderMetrics(metrics) {
  const container = document.getElementById('metrics');
  const items = [
    ['R²', metrics.r2],
    ['MAE', metrics.mae],
    ['MSE', metrics.mse],
    ['RMSE', metrics.rmse],
  ];

  container.innerHTML = items
    .map(
      ([label, value]) => `
        <div class="metric">
          <span class="metric-label">${label}</span>
          <span class="metric-value">${Number(value).toFixed(4)}</span>
        </div>
      `
    )
    .join('');
}

function renderCoefficients(featureNames, coefficients) {
  const container = document.getElementById('coefficients');

  const html = featureNames
    .map((name, idx) => `
      <div class="coefficient-item">
        <div class="coefficient-name">${window.DatasetUtils.escapeHTML(name)}</div>
        <div class="coefficient-value">${Number(coefficients[idx]).toFixed(4)}</div>
      </div>
    `)
    .join('');

  container.innerHTML = html;
}

function renderScatterChart(yTrue, yPred, config) {
  const ctx = document.getElementById('scatterChart');
  if (scatterChartInstance) scatterChartInstance.destroy();
  const minValue = Math.min(...yTrue, ...yPred);
  const maxValue = Math.max(...yTrue, ...yPred);

  scatterChartInstance = new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: 'Actual vs Predicted',
          data: yTrue.map((value, index) => ({ x: value, y: yPred[index] })),
          pointRadius: 4,
          pointBackgroundColor: '#0f4c81',
        },
        {
          label: 'Ideal fit',
          data: [
            { x: minValue, y: minValue },
            { x: maxValue, y: maxValue },
          ],
          type: 'line',
          borderColor: '#f39a5d',
          borderWidth: 2,
          pointRadius: 0,
          showLine: true,
        },
      ],
    },
    options: {
      responsive: true,
      scales: {
        x: {
          title: { display: true, text: `Actual ${config.targetLabel} (${config.targetUnit})` },
        },
        y: {
          title: { display: true, text: `Predicted ${config.targetLabel} (${config.targetUnit})` },
        },
      },
      plugins: {
        legend: { display: true },
      },
    },
  });
}

async function loadData(datasetId = 'startups') {
  const requestId = ++activeLoadRequest;
  const config = window.DATASET_CONFIGS[datasetId];
  const status = document.getElementById('selectionStatus');
  if (status) status.textContent = `Loading ${config.label}…`;

  try {
    let rawRows;
    if (config.rows) {
      rawRows = config.rows.map((row) => ({ ...row }));
    } else {
      const response = await fetch(config.file);
      if (!response.ok) {
        throw new Error(`Failed to fetch dataset: ${response.status}`);
      }
      rawRows = window.DatasetUtils.parseCSV(await response.text());
    }
    if (requestId !== activeLoadRequest) return;
    const appendedRows = window.DatasetImport?.getAppendedRows(datasetId) || [];
    rawRows = rawRows.concat(appendedRows);
    const { rows, features } = window.DatasetUtils.prepareRows(rawRows, config);
    document.getElementById('pageTitle').textContent = config.title;
    document.getElementById('pageSubtitle').textContent = config.subtitle;
    document.getElementById('scatterTitle').textContent = `Actual vs Predicted ${config.targetLabel}`;
    document.getElementById('predictionLabel').textContent = `Predicted ${config.targetLabel.toLowerCase()}`;
    document.getElementById('scoreCaption').textContent = `Held-out test set ${config.targetLabel.toLowerCase()} model performance.`;
    document.title = `${config.label} Multiple Linear Regression`;
    document.getElementById('dataPreview').innerHTML = rowsToTable(rawRows, 8);
    const datasetNote = document.getElementById('datasetNote');
    datasetNote.textContent = config.note || '';
    datasetNote.hidden = !config.note;

    const { coefficients, featureNames, metrics, yPred, yTrue, split } = fitLinearRegression(
      rows,
      features,
      config.target,
    );
    renderMetrics(metrics);
    renderCoefficients(featureNames, coefficients);
    renderScatterChart(yTrue, yPred, config);

    const heroR2 = document.getElementById('heroR2');
    if (heroR2) heroR2.textContent = Number(metrics.r2).toFixed(4);
    window.dispatchEvent(new CustomEvent('datasetchange', {
      detail: { config, rows, rawRows, features, split },
    }));
  } catch (error) {
    if (requestId !== activeLoadRequest) return;
    document.getElementById('dataPreview').innerHTML = `<p style="color: red;">${error.message}</p>`;
    if (status) status.textContent = error.message;
    console.error(error);
  }
}

window.addEventListener('DOMContentLoaded', () => {
  const selector = document.getElementById('datasetSelect');
  selector.addEventListener('change', () => loadData(selector.value));
  loadData(selector.value);
});

window.loadDataset = loadData;
