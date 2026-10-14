let predictorNames = [];
let featureDefinitions = [];
let featureByKey = new Map();
let targetKey = '';
let datasetConfig;
let maxFeatureCount = 0;
let cvScoreCache = new Map();
const selectionMethods = [
  { id: 'lasso', label: 'Lasso', detail: 'L1 ranking' },
  { id: 'forward', label: 'Forward SFS', detail: 'Greedy' },
  { id: 'rfe', label: 'RFE', detail: 'Backward' },
  { id: 'elasticNet', label: 'ElasticNet', detail: 'L1 + L2' },
  { id: 'exhaustive', label: 'Exhaustive Search', detail: 'Best subset' },
];

let selectionState;
let rmseChart;
let r2Chart;

function targetValue(row) {
  return row[targetKey];
}

function featureLabel(key) {
  return featureByKey.get(key)?.label || key;
}

function seededRandom(seed) {
  let value = seed >>> 0;
  return () => {
    value += 0x6d2b79f5;
    let mixed = value;
    mixed = Math.imul(mixed ^ (mixed >>> 15), mixed | 1);
    mixed ^= mixed + Math.imul(mixed ^ (mixed >>> 7), mixed | 61);
    return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296;
  };
}

function splitRows(length) {
  const indices = Array.from({ length }, (_, index) => index);
  const random = seededRandom(42);
  for (let index = indices.length - 1; index > 0; index--) {
    const swapIndex = Math.floor(random() * (index + 1));
    [indices[index], indices[swapIndex]] = [indices[swapIndex], indices[index]];
  }
  const testCount = Math.ceil(length * 0.2);
  return {
    test: indices.slice(0, testCount),
    train: indices.slice(testCount),
  };
}

function solveLinearSystem(matrix, vector) {
  const size = matrix.length;
  const augmented = matrix.map((row, index) => [...row, vector[index]]);

  for (let column = 0; column < size; column++) {
    let pivot = column;
    for (let row = column + 1; row < size; row++) {
      if (Math.abs(augmented[row][column]) > Math.abs(augmented[pivot][column])) pivot = row;
    }
    if (Math.abs(augmented[pivot][column]) < 1e-10) {
      throw new Error('The selected feature matrix is singular.');
    }
    [augmented[column], augmented[pivot]] = [augmented[pivot], augmented[column]];

    const divisor = augmented[column][column];
    for (let index = column; index <= size; index++) augmented[column][index] /= divisor;
    for (let row = 0; row < size; row++) {
      if (row === column) continue;
      const factor = augmented[row][column];
      for (let index = column; index <= size; index++) {
        augmented[row][index] -= factor * augmented[column][index];
      }
    }
  }

  return augmented.map((row) => row[size]);
}

function fitOls(rows, indices, features) {
  const dimension = features.length + 1;
  const xtx = Array.from({ length: dimension }, () => Array(dimension).fill(0));
  const xty = Array(dimension).fill(0);

  indices.forEach((rowIndex) => {
    const row = rows[rowIndex];
    const values = [1, ...features.map((feature) => row[feature])];
    values.forEach((value, left) => {
      xty[left] += value * targetValue(row);
      values.forEach((other, right) => {
        xtx[left][right] += value * other;
      });
    });
  });

  return { coefficients: solveLinearSystem(xtx, xty), xtx };
}

function predictRow(row, features, coefficients) {
  return coefficients[0] + features.reduce(
    (sum, feature, index) => sum + row[feature] * coefficients[index + 1],
    0,
  );
}

function rootMeanSquaredError(actual, predicted) {
  const squaredError = actual.reduce((sum, value, index) => sum + (value - predicted[index]) ** 2, 0);
  return Math.sqrt(squaredError / actual.length);
}

function crossValidatedRmse(rows, trainIndices, features) {
  const cacheKey = [...features].sort().join('|');
  if (cvScoreCache.has(cacheKey)) return cvScoreCache.get(cacheKey);
  const foldCount = 5;
  const folds = Array.from({ length: foldCount }, () => []);
  trainIndices.forEach((rowIndex, index) => folds[index % foldCount].push(rowIndex));

  const actual = [];
  const predicted = [];
  folds.forEach((validationIndices) => {
    const validationSet = new Set(validationIndices);
    const fittingIndices = trainIndices.filter((rowIndex) => !validationSet.has(rowIndex));
    const model = fitOls(rows, fittingIndices, features);
    validationIndices.forEach((rowIndex) => {
      actual.push(targetValue(rows[rowIndex]));
      predicted.push(predictRow(rows[rowIndex], features, model.coefficients));
    });
  });
  const score = rootMeanSquaredError(actual, predicted);
  cvScoreCache.set(cacheKey, score);
  return score;
}

function getCombinations(features, count, start = 0, prefix = [], output = []) {
  if (prefix.length === count) {
    output.push(prefix);
    return output;
  }
  for (let index = start; index <= features.length - (count - prefix.length); index++) {
    getCombinations(features, count, index + 1, [...prefix, features[index]], output);
  }
  return output;
}

function chooseLowestCv(rows, trainIndices, candidates) {
  return candidates
    .map((features) => ({ features, score: crossValidatedRmse(rows, trainIndices, features) }))
    .sort((left, right) => left.score - right.score)[0].features;
}

function forwardSelection(rows, trainIndices) {
  const selected = [];
  const subsets = [];
  while (selected.length < maxFeatureCount) {
    const remaining = predictorNames.filter((feature) => !selected.includes(feature));
    const next = chooseLowestCv(rows, trainIndices, remaining.map((feature) => [...selected, feature]));
    selected.push(next[selected.length]);
    subsets.push([...selected]);
  }
  return subsets;
}

function standardizedCoefficientRanking(rows, trainIndices, alpha, l1Ratio) {
  const means = predictorNames.map((feature) => trainIndices.reduce(
    (sum, rowIndex) => sum + rows[rowIndex][feature],
    0,
  ) / trainIndices.length);
  const deviations = predictorNames.map((feature, index) => Math.sqrt(
    trainIndices.reduce((sum, rowIndex) => sum + (rows[rowIndex][feature] - means[index]) ** 2, 0)
      / trainIndices.length,
  ) || 1);
  const targetMean = trainIndices.reduce((sum, rowIndex) => sum + targetValue(rows[rowIndex]), 0)
    / trainIndices.length;
  const targetDeviation = Math.sqrt(
    trainIndices.reduce((sum, rowIndex) => sum + (targetValue(rows[rowIndex]) - targetMean) ** 2, 0)
      / trainIndices.length,
  ) || 1;
  const x = trainIndices.map((rowIndex) => predictorNames.map(
    (feature, index) => (rows[rowIndex][feature] - means[index]) / deviations[index],
  ));
  const y = trainIndices.map((rowIndex) => (targetValue(rows[rowIndex]) - targetMean) / targetDeviation);
  const weights = Array(predictorNames.length).fill(0);

  for (let iteration = 0; iteration < 1200; iteration++) {
    let largestChange = 0;
    for (let featureIndex = 0; featureIndex < weights.length; featureIndex++) {
      let partialCorrelation = 0;
      let squaredNorm = 0;
      x.forEach((row, rowIndex) => {
        const currentPrediction = row.reduce((sum, value, index) => sum + value * weights[index], 0);
        partialCorrelation += row[featureIndex] * (y[rowIndex] - currentPrediction + row[featureIndex] * weights[featureIndex]);
        squaredNorm += row[featureIndex] ** 2;
      });
      const threshold = alpha * l1Ratio * trainIndices.length;
      const shrunk = Math.sign(partialCorrelation) * Math.max(0, Math.abs(partialCorrelation) - threshold);
      const updated = shrunk / (squaredNorm + alpha * (1 - l1Ratio) * trainIndices.length);
      largestChange = Math.max(largestChange, Math.abs(updated - weights[featureIndex]));
      weights[featureIndex] = updated;
    }
    if (largestChange < 1e-8) break;
  }

  return predictorNames
    .map((feature, index) => ({ feature, weight: Math.abs(weights[index]) }))
    .sort((left, right) => right.weight - left.weight)
    .map((item) => item.feature);
}

function subsetsFromRanking(ranking) {
  return Array.from({ length: maxFeatureCount }, (_, index) => ranking.slice(0, index + 1));
}

function recursiveFeatureElimination(rows, trainIndices) {
  const remaining = [...predictorNames];
  const subsets = Array(maxFeatureCount);
  const targetMean = trainIndices.reduce((sum, rowIndex) => sum + targetValue(rows[rowIndex]), 0)
    / trainIndices.length;
  const targetDeviation = Math.sqrt(trainIndices.reduce(
    (sum, rowIndex) => sum + (targetValue(rows[rowIndex]) - targetMean) ** 2,
    0,
  ) / trainIndices.length) || 1;
  const removeWeakestFeature = () => {
    const { coefficients } = fitOls(rows, trainIndices, remaining);
    const standardizedMagnitudes = remaining.map((feature, index) => {
      const mean = trainIndices.reduce((sum, rowIndex) => sum + rows[rowIndex][feature], 0)
        / trainIndices.length;
      const deviation = Math.sqrt(trainIndices.reduce(
        (sum, rowIndex) => sum + (rows[rowIndex][feature] - mean) ** 2,
        0,
      ) / trainIndices.length) || 1;
      return Math.abs(coefficients[index + 1] * deviation / targetDeviation);
    });
    const weakestIndex = standardizedMagnitudes.reduce(
      (weakest, value, index, values) => value < values[weakest] ? index : weakest,
      0,
    );
    remaining.splice(weakestIndex, 1);
  };
  while (remaining.length > maxFeatureCount) removeWeakestFeature();
  for (let count = maxFeatureCount; count > 0; count--) {
    subsets[count - 1] = [...remaining];
    if (count > 1) removeWeakestFeature();
  }
  return subsets;
}

function buildMethodSubsets(rows, trainIndices) {
  const forward = forwardSelection(rows, trainIndices);
  const exhaustive = Array.from({ length: maxFeatureCount }, (_, index) => {
    const candidates = getCombinations(predictorNames, index + 1);
    return chooseLowestCv(rows, trainIndices, candidates);
  });
  return {
    lasso: subsetsFromRanking(standardizedCoefficientRanking(rows, trainIndices, 0.05, 1)),
    forward,
    rfe: recursiveFeatureElimination(rows, trainIndices),
    elasticNet: subsetsFromRanking(standardizedCoefficientRanking(rows, trainIndices, 0.05, 0.5)),
    exhaustive,
  };
}

function calculateMetrics(rows, trainIndices, testIndices, features) {
  const model = fitOls(rows, trainIndices, features);
  const actual = testIndices.map((rowIndex) => targetValue(rows[rowIndex]));
  const predicted = testIndices.map((rowIndex) => predictRow(rows[rowIndex], features, model.coefficients));
  const mean = actual.reduce((sum, value) => sum + value, 0) / actual.length;
  const residualSum = actual.reduce((sum, value, index) => sum + (value - predicted[index]) ** 2, 0);
  const totalSum = actual.reduce((sum, value) => sum + (value - mean) ** 2, 0);
  const r2 = 1 - residualSum / totalSum;
  const mae = actual.reduce((sum, value, index) => sum + Math.abs(value - predicted[index]), 0) / actual.length;
  const adjustedR2 = 1 - (1 - r2) * (actual.length - 1) / Math.max(1, actual.length - features.length - 1);
  const trainActual = trainIndices.map((rowIndex) => targetValue(rows[rowIndex]));
  const trainPredicted = trainIndices.map((rowIndex) => predictRow(rows[rowIndex], features, model.coefficients));
  const trainMean = trainActual.reduce((sum, value) => sum + value, 0) / trainActual.length;
  const trainTotalSum = trainActual.reduce((sum, value) => sum + (value - trainMean) ** 2, 0);
  const trainResidualSum = trainActual.reduce(
    (sum, value, index) => sum + (value - trainPredicted[index]) ** 2,
    0,
  );
  const trainR2 = 1 - trainResidualSum / trainTotalSum;
  const residualDegreesOfFreedom = trainIndices.length - features.length - 1;
  const fStatistic = (trainR2 / features.length)
    / Math.max(1e-12, (1 - trainR2) / residualDegreesOfFreedom);
  return {
    ...model,
    actual,
    predicted,
    residuals: actual.map((value, index) => value - predicted[index]),
    rmse: rootMeanSquaredError(actual, predicted),
    mae,
    r2,
    adjustedR2,
    trainR2,
    fStatistic,
  };
}

function invertMatrix(matrix) {
  return matrix.map((_, column) => solveLinearSystem(matrix, matrix.map((__, index) => index === column ? 1 : 0)));
}

function normalTailProbability(value) {
  const absolute = Math.abs(value) / Math.sqrt(2);
  const t = 1 / (1 + 0.3275911 * absolute);
  const erf = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
    - 0.284496736) * t + 0.254829592) * t * Math.exp(-absolute * absolute);
  return 1 - erf;
}

function coefficientStatistics(result, trainCount) {
  const inverse = invertMatrix(result.xtx);
  const trainPrediction = selectionState.train.map((rowIndex) => (
    predictRow(selectionState.rows[rowIndex], selectionState.features, result.coefficients)
  ));
  const trainActual = selectionState.train.map((rowIndex) => targetValue(selectionState.rows[rowIndex]));
  const residualVariance = trainActual.reduce(
    (sum, value, index) => sum + (value - trainPrediction[index]) ** 2,
    0,
  ) / Math.max(1, trainCount - result.coefficients.length);
  return result.coefficients.map((estimate, index) => {
    const standardError = Math.sqrt(Math.max(0, inverse[index][index] * residualVariance));
    const tStatistic = standardError ? estimate / standardError : 0;
    return {
      name: index === 0 ? 'Intercept' : selectionState.features[index - 1],
      estimate,
      standardError,
      tStatistic,
      pValue: normalTailProbability(tStatistic),
    };
  });
}

function formatTarget(value, digits = 2) {
  return window.DatasetUtils.targetText(value, datasetConfig, digits);
}

function renderMethodButtons() {
  const container = document.getElementById('selectionMethods');
  container.innerHTML = selectionMethods.map((method) => `
    <button class="selection-method" type="button" data-method="${method.id}" aria-pressed="false">
      <span>${method.label}</span><small>${method.detail}</small>
    </button>
  `).join('') + '<button class="selection-method compare-method" type="button" data-method="compare" aria-pressed="false">Compare all methods</button>';
}

function renderFeatureCountButtons() {
  const container = document.getElementById('featureCounts');
  container.innerHTML = Array.from({ length: maxFeatureCount }, (_, index) => {
    const count = index + 1;
    return `<button class="feature-count" type="button" data-count="${count}" aria-pressed="false">k = ${count}<small>${count} predictor${count === 1 ? '' : 's'}</small></button>`;
  }).join('');
}

function renderCharts() {
  if (!selectionState || !window.Chart) return;
  const methodsToShow = selectionState.method === 'compare'
    ? selectionMethods
    : selectionMethods.filter((method) => method.id === selectionState.method);
  const palette = ['#166b85', '#e4863b', '#338568', '#b34f4c', '#785a9f'];
  const commonOptions = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'nearest', intersect: false },
    plugins: { legend: { position: 'bottom', labels: { usePointStyle: true, boxWidth: 8 } } },
  };
  const datasetsFor = (key) => methodsToShow.map((method) => ({
    label: method.label,
    data: Array.from({ length: maxFeatureCount }, (_, index) => {
      const result = selectionState.results[method.id][index];
      return key === 'rmse' ? result.rmse : result.r2;
    }),
    borderColor: palette[selectionMethods.findIndex((item) => item.id === method.id)],
    backgroundColor: palette[selectionMethods.findIndex((item) => item.id === method.id)],
    tension: 0.25,
    pointRadius: 4,
  }));

  if (rmseChart) rmseChart.destroy();
  if (r2Chart) r2Chart.destroy();
  rmseChart = new Chart(document.getElementById('selectionRmseChart'), {
    type: 'line',
    data: { labels: Array.from({ length: maxFeatureCount }, (_, index) => `k = ${index + 1}`), datasets: datasetsFor('rmse') },
    options: { ...commonOptions, scales: { y: { beginAtZero: false, title: { display: true, text: `Test RMSE (${datasetConfig.targetUnit})` } } } },
  });
  r2Chart = new Chart(document.getElementById('selectionR2Chart'), {
    type: 'line',
    data: { labels: Array.from({ length: maxFeatureCount }, (_, index) => `k = ${index + 1}`), datasets: datasetsFor('r2') },
    options: { ...commonOptions, scales: { y: { title: { display: true, text: 'Test R²' } } } },
  });
}

function renderCoefficientTable(result) {
  const rows = coefficientStatistics(result, selectionState.train.length);
  document.getElementById('selectionCoefficients').innerHTML = `
    <table>
      <thead><tr><th>Predictor</th><th>Estimate</th><th>Std. error</th><th>t-stat</th><th>p-value*</th><th>Approx. 95% CI</th></tr></thead>
      <tbody>${rows.map((row) => {
        const significant = row.pValue < 0.05;
        return `<tr><th scope="row">${window.DatasetUtils.escapeHTML(row.name)}</th><td>${row.estimate.toFixed(4)}</td><td>${row.standardError.toFixed(2)}</td><td>${row.tStatistic.toFixed(3)}</td><td>${row.pValue < 0.0001 ? '&lt; 0.0001' : row.pValue.toFixed(4)} ${significant ? 'Significant' : 'n.s.'}</td><td>[${(row.estimate - 1.96 * row.standardError).toFixed(3)}, ${(row.estimate + 1.96 * row.standardError).toFixed(3)}]</td></tr>`;
      }).join('')}</tbody>
    </table>
    <p class="selection-note">*Two-sided normal approximation; confidence intervals are approximate.</p>
  `;
}

function renderResiduals(result) {
  const residualRows = selectionState.test.map((rowIndex, index) => ({
    id: rowIndex + 1,
    actual: result.actual[index],
    predicted: result.predicted[index],
    residual: result.residuals[index],
  })).sort((left, right) => Math.abs(right.residual) - Math.abs(left.residual));
  document.getElementById('selectionResiduals').innerHTML = `
    <table>
      <thead><tr><th>Rank</th><th>Sample</th><th>Actual ${window.DatasetUtils.escapeHTML(datasetConfig.targetLabel)}</th><th>Predicted ${window.DatasetUtils.escapeHTML(datasetConfig.targetLabel)}</th><th>Residual</th><th>Absolute error</th><th>Status</th></tr></thead>
      <tbody>${residualRows.map((row, index) => `<tr>
        <td>${index + 1}</td><td>${datasetConfig.sampleLabel} #${row.id}</td><td>${formatTarget(row.actual)}</td>
        <td>${formatTarget(row.predicted)}</td><td>${formatTarget(row.residual)}</td>
        <td>${formatTarget(Math.abs(row.residual))}</td><td><span class="residual-status ${Math.abs(row.residual) > result.rmse ? 'high' : ''}">${Math.abs(row.residual) > result.rmse ? 'High residual' : 'In range'}</span></td>
      </tr>`).join('')}</tbody>
    </table>
  `;
}

function currentFeatures() {
  const method = selectionState.method === 'compare' ? 'exhaustive' : selectionState.method;
  return selectionState.subsets[method][selectionState.count - 1];
}

function renderSelection() {
  const features = currentFeatures();
  selectionState.features = features;
  const method = selectionState.method === 'compare' ? 'exhaustive' : selectionState.method;
  const result = selectionState.results[method][selectionState.count - 1];
  document.querySelectorAll('.selection-method').forEach((button) => {
    button.setAttribute('aria-pressed', String(button.dataset.method === selectionState.method));
  });
  document.querySelectorAll('.feature-count').forEach((button) => {
    button.setAttribute('aria-pressed', String(Number(button.dataset.count) === selectionState.count));
  });

  document.getElementById('selectionRmse').textContent = formatTarget(result.rmse);
  document.getElementById('selectionR2').textContent = result.r2.toFixed(4);
  document.getElementById('selectionAdjustedR2').textContent = result.adjustedR2.toFixed(4);
  document.getElementById('selectionMae').textContent = formatTarget(result.mae);
  document.getElementById('selectionFStatistic').textContent = result.fStatistic.toFixed(2);
  document.getElementById('selectionActiveFeatures').innerHTML = features
    .map((feature) => `<span class="active-feature">${window.DatasetUtils.escapeHTML(featureLabel(feature))}</span>`).join('');
  document.getElementById('selectionEquation').textContent = `${datasetConfig.targetLabel} = ${result.coefficients[0].toFixed(4)}${features.map((feature, index) => ` ${result.coefficients[index + 1] < 0 ? '−' : '+'} ${Math.abs(result.coefficients[index + 1]).toFixed(4)} × ${featureLabel(feature)}`).join('')}`;
  renderCoefficientTable(result);
  renderResiduals(result);
  renderCharts();
  updatePrediction();
}

function updatePrediction() {
  if (!selectionState) return;
  const values = {};
  document.querySelectorAll('[data-predictor]').forEach((input) => {
    values[input.dataset.predictor] = Number(input.value);
  });
  datasetConfig.features.filter((feature) => feature.type === 'categorical').forEach((definition) => {
    const categoryControl = [...document.querySelectorAll('[data-category]')]
      .find((control) => control.dataset.category === definition.key);
    const selectedLevel = categoryControl.value;
    featureDefinitions.filter((feature) => feature.type === 'dummy' && feature.source === definition.key)
      .forEach((feature) => {
        values[feature.key] = Number(selectedLevel === feature.level);
      });
  });
  const method = selectionState.method === 'compare' ? 'exhaustive' : selectionState.method;
  const result = selectionState.results[method][selectionState.count - 1];
  const predicted = predictRow(values, selectionState.features, result.coefficients);
  document.getElementById('predictedProfit').textContent = formatTarget(predicted);
  document.getElementById('predictionFeatures').textContent = `Using ${selectionState.features.length} active predictor${selectionState.features.length === 1 ? '' : 's'}: ${selectionState.features.map(featureLabel).join(', ')}`;
}

function renderPredictorControls() {
  const numericControls = datasetConfig.features.filter((feature) => feature.type === 'numeric');
  const numericMarkup = numericControls.map((definition) => {
    const values = selectionState.rows.map((row) => row[definition.key]);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const step = Math.max((max - min) / 1000, 0.001);
    const mean = values.reduce((sum, item) => sum + item, 0) / values.length;
    const value = Number((min + Math.round((mean - min) / step) * step).toFixed(6));
    return `
    <label class="predictor-control">
      <span>${window.DatasetUtils.escapeHTML(definition.label)} <output data-output="${window.DatasetUtils.escapeHTML(definition.key)}">${value.toFixed(3)}</output></span>
      <input type="number" data-predictor="${window.DatasetUtils.escapeHTML(definition.key)}" min="${min}" max="${max}" step="${step}" value="${value}" aria-label="${window.DatasetUtils.escapeHTML(definition.label)}">
      <input type="range" data-predictor="${window.DatasetUtils.escapeHTML(definition.key)}" min="${min}" max="${max}" step="${step}" value="${value}" aria-label="Adjust ${window.DatasetUtils.escapeHTML(definition.label)}">
    </label>
  `;
  }).join('');
  const categoryMarkup = datasetConfig.features.filter((feature) => feature.type === 'categorical')
    .map((definition) => {
      const levels = [...new Set(selectionState.rawRows.map((row) => row[definition.key]))];
      return `<label class="state-control">${window.DatasetUtils.escapeHTML(definition.label)}
        <select data-category="${window.DatasetUtils.escapeHTML(definition.key)}" aria-label="Choose ${window.DatasetUtils.escapeHTML(definition.label)}">
          ${levels.map((level) => `<option value="${window.DatasetUtils.escapeHTML(level)}" ${String(level) === String(definition.baseline) ? 'selected' : ''}>${window.DatasetUtils.escapeHTML(level)}</option>`).join('')}
        </select>
      </label>`;
    }).join('');
  document.getElementById('predictorControls').innerHTML = numericMarkup + categoryMarkup;
}

function bindFeatureSelectionControls() {
  document.getElementById('selectionMethods').addEventListener('click', (event) => {
    const button = event.target.closest('[data-method]');
    if (!button || !selectionState) return;
    selectionState.method = button.dataset.method;
    renderSelection();
  });
  document.getElementById('featureCounts').addEventListener('click', (event) => {
    const button = event.target.closest('[data-count]');
    if (!button || !selectionState) return;
    selectionState.count = Number(button.dataset.count);
    renderSelection();
  });
  document.getElementById('predictorControls').addEventListener('input', (event) => {
    if (!event.target.matches('[data-predictor]')) return;
    document.querySelectorAll('[data-predictor]').forEach((input) => {
      if (input !== event.target && input.dataset.predictor === event.target.dataset.predictor) {
        input.value = event.target.value;
      }
    });
    const output = [...document.querySelectorAll('[data-output]')]
      .find((item) => item.dataset.output === event.target.dataset.predictor);
    if (output) output.textContent = Number(event.target.value).toFixed(3);
    updatePrediction();
  });
  document.getElementById('predictorControls').addEventListener('change', (event) => {
    if (event.target.matches('[data-category]')) updatePrediction();
  });
}

function loadFeatureSelection(event) {
  const { config, rows, rawRows, features, split } = event.detail;
  datasetConfig = config;
  targetKey = config.target;
  featureDefinitions = features;
  featureByKey = new Map(features.map((feature) => [feature.key, feature]));
  predictorNames = features.map((feature) => feature.key);
  maxFeatureCount = Math.min(5, predictorNames.length);
  cvScoreCache = new Map();
  try {
    const subsets = buildMethodSubsets(rows, split.train);
    const results = Object.fromEntries(selectionMethods.map((method) => [
      method.id,
      subsets[method.id].map((features) => calculateMetrics(rows, split.train, split.test, features)),
    ]));
    selectionState = {
      rows,
      rawRows,
      train: split.train,
      test: split.test,
      subsets,
      results,
      method: 'lasso',
      maxFeatures: maxFeatureCount,
      count: Math.min(2, maxFeatureCount),
      features: subsets.lasso[1],
    };
    renderMethodButtons();
    renderFeatureCountButtons();
    renderPredictorControls();
    renderSelection();
    document.getElementById('selectionStatus').textContent = `${rows.length} rows · ${split.train.length} train · ${split.test.length} test · 5-fold training CV`;
    document.querySelector('.selection-intro').textContent = `Compare five feature-selection strategies across up to ${maxFeatureCount} predictors. Selection uses five-fold cross-validation on training data; final metrics use the held-out test split.`;
    if (!selectionControlsBound) {
      bindFeatureSelectionControls();
      selectionControlsBound = true;
    }
  } catch (error) {
    document.getElementById('selectionStatus').textContent = error.message;
    console.error(error);
  }
}

let selectionControlsBound = false;
window.addEventListener('datasetchange', loadFeatureSelection);