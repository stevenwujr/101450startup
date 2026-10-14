const csvUrl = './50_Startups.csv';

function parseCSV(text) {
  const lines = text.trim().split(/\r?\n/);
  const headers = lines[0].split(',');
  const rows = [];

  for (let i = 1; i < lines.length; i++) {
    if (!lines[i].trim()) continue;
    const values = lines[i].split(',');
    const row = {};
    headers.forEach((header, idx) => {
      const raw = values[idx] ? values[idx].trim() : '';
      const numericVal = Number(raw);
      row[header] = Number.isFinite(numericVal) ? numericVal : raw;
    });
    rows.push(row);
  }

  return rows;
}

function rowsToTable(rows, maxRows = 8) {
  const columns = Object.keys(rows[0] || {});
  const limited = rows.slice(0, maxRows);

  let html = '<table><thead><tr>';
  columns.forEach((col) => {
    html += `<th>${col}</th>`;
  });
  html += '</tr></thead><tbody>';

  limited.forEach((row) => {
    html += '<tr>';
    columns.forEach((col) => {
      const value = row[col];
      html += `<td>${typeof value === 'number' ? value.toLocaleString(undefined, { maximumFractionDigits: 2 }) : value}</td>`;
    });
    html += '</tr>';
  });
  html += '</tbody></table>';

  return html;
}

function prepareMatrix(rows) {
  const states = [...new Set(rows.map((r) => r.State))];
  const baseState = states[0];

  const featureNames = [
    'Intercept',
    'R&D Spend',
    'Administration',
    'Marketing Spend',
    ...states.filter((s) => s !== baseState).map((s) => `State_${s}`),
  ];

  const X = [];
  const y = [];

  rows.forEach((row) => {
    const featureVector = [1, row['R&D Spend'], row['Administration'], row['Marketing Spend']];
    states.filter((s) => s !== baseState).forEach((stateName) => {
      featureVector.push(row.State === stateName ? 1 : 0);
    });
    X.push(featureVector);
    y.push(row.Profit);
  });

  return { X, y, featureNames };
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

function fitLinearRegression(rows) {
  const { X, y, featureNames } = prepareMatrix(rows);
  const Xt = transpose(X);
  const XtX = multiplyMatrix(Xt, X);
  const Xty = multiplyMatrix(Xt, y.map((value) => [value]));
  const b = Xty.map((row) => row[0]);
  const coefficients = solveLinearSystem(XtX, b);

  const yPred = X.map((row) =>
    row.reduce((sum, value, i) => sum + value * coefficients[i], 0)
  );

  const metrics = computeMetrics(y, yPred);
  return { coefficients, featureNames, metrics, yPred, yTrue: y };
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
        <div class="coefficient-name">${name}</div>
        <div class="coefficient-value">${Number(coefficients[idx]).toFixed(4)}</div>
      </div>
    `)
    .join('');

  container.innerHTML = html;
}

function renderScatterChart(yTrue, yPred) {
  const ctx = document.getElementById('scatterChart');

  new Chart(ctx, {
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
            { x: Math.min(...yTrue), y: Math.min(...yTrue) },
            { x: Math.max(...yTrue), y: Math.max(...yTrue) },
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
          title: { display: true, text: 'Actual Profit' },
        },
        y: {
          title: { display: true, text: 'Predicted Profit' },
        },
      },
      plugins: {
        legend: { display: true },
      },
    },
  });
}

async function loadData() {
  try {
    const response = await fetch(csvUrl);
    if (!response.ok) {
      throw new Error(`Failed to fetch dataset: ${response.status}`);
    }

    const text = await response.text();
    const rows = parseCSV(text);
    document.getElementById('dataPreview').innerHTML = rowsToTable(rows, 8);

    const { coefficients, featureNames, metrics, yPred, yTrue } = fitLinearRegression(rows);
    renderMetrics(metrics);
    renderCoefficients(featureNames, coefficients);
    renderScatterChart(yTrue, yPred);

    const heroR2 = document.getElementById('heroR2');
    if (heroR2) heroR2.textContent = Number(metrics.r2).toFixed(4);
  } catch (error) {
    document.getElementById('dataPreview').innerHTML = `<p style="color: red;">${error.message}</p>`;
    console.error(error);
  }
}

window.addEventListener('DOMContentLoaded', loadData);
