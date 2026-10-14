const customDatasetStorageKey = 'startup-regression:custom-datasets';
const appendedRowsStoragePrefix = 'startup-regression:appended:';

let importedCsvRows = null;
let importFileName = '';

function importStatus(message, state = '') {
  const status = document.getElementById('importStatus');
  status.textContent = message;
  status.dataset.state = state;
}

function storedJson(key, fallback) {
  try {
    const value = localStorage.getItem(key);
    return value ? JSON.parse(value) : fallback;
  } catch {
    return fallback;
  }
}

function datasetOption(config) {
  const option = document.createElement('option');
  option.value = config.id;
  option.textContent = config.label;
  return option;
}

function restoreCustomDatasets() {
  const records = storedJson(customDatasetStorageKey, []);
  const selector = document.getElementById('datasetSelect');
  records.forEach((record) => {
    if (!record?.config?.id || !Array.isArray(record.rows)) return;
    const config = { ...record.config, rows: record.rows };
    window.DATASET_CONFIGS[config.id] = config;
    selector.append(datasetOption(config));
  });
  const lastDataset = localStorage.getItem('startup-regression:active-dataset');
  if (lastDataset && window.DATASET_CONFIGS[lastDataset]) {
    selector.value = lastDataset;
    window.loadDataset(lastDataset);
  }
}

function setImportMode() {
  const mode = document.getElementById('datasetImportMode').value;
  const isNew = mode === 'new';
  document.getElementById('newDatasetNameField').hidden = !isNew;
  document.getElementById('targetColumnField').hidden = !isNew;
  document.getElementById('targetUnitField').hidden = !isNew;
  validateImportedCsv();
}

function validateNewDataset() {
  if (!importedCsvRows?.length) return false;
  const target = document.getElementById('datasetTargetColumn').value;
  if (!target) {
    importStatus('Choose a numeric column to predict.', 'error');
    return false;
  }
  try {
    const features = window.DatasetUtils.inferFeatures(importedCsvRows, target);
    const name = document.getElementById('newDatasetName').value.trim();
    if (!name) {
      importStatus('Enter a name for the new dataset.', 'error');
      return false;
    }
    importStatus(`${importedCsvRows.length} rows · ${features.length} predictor columns inferred · target: ${target}.`, 'success');
    return true;
  } catch (error) {
    importStatus(error.message, 'error');
    return false;
  }
}

function validateAppendDataset() {
  if (!importedCsvRows?.length) return false;
  const datasetId = document.getElementById('datasetSelect').value;
  const config = window.DATASET_CONFIGS[datasetId];
  if (!config) {
    importStatus('Choose an available dataset before appending rows.', 'error');
    return false;
  }
  try {
    const rows = window.DatasetUtils.validateRows(importedCsvRows, config);
    importStatus(`${rows.length} rows match ${config.label} and are ready to append.`, 'success');
    return true;
  } catch (error) {
    importStatus(error.message, 'error');
    return false;
  }
}

function validateImportedCsv() {
  const submit = document.getElementById('confirmDatasetImport');
  if (!importedCsvRows) {
    submit.disabled = true;
    return false;
  }
  const valid = document.getElementById('datasetImportMode').value === 'new'
    ? validateNewDataset()
    : validateAppendDataset();
  submit.disabled = !valid;
  return valid;
}

function targetDefault(numericColumns) {
  const commonTarget = numericColumns.find((column) => /^(target|y|medv|price|profit|saleprice)$/i.test(column));
  return commonTarget || numericColumns[numericColumns.length - 1] || '';
}

async function readImportFile(file) {
  importedCsvRows = null;
  importFileName = file.name;
  document.getElementById('confirmDatasetImport').disabled = true;
  importStatus('Reading CSV…');
  try {
    if (file.size > 1_500_000) throw new Error('CSV files must be 1.5 MB or smaller for browser-local storage.');
    const rows = window.DatasetUtils.parseCSV(await file.text());
    if (rows.length < 2) throw new Error('Add at least two data rows to the CSV.');
    importedCsvRows = rows;
    const columns = window.DatasetUtils.numericColumns(rows);
    const targetSelect = document.getElementById('datasetTargetColumn');
    targetSelect.replaceChildren(...columns.map((column) => {
      const option = document.createElement('option');
      option.value = column;
      option.textContent = column;
      return option;
    }));
    targetSelect.value = targetDefault(columns);
    targetSelect.disabled = columns.length === 0;
    if (columns.length === 0 && document.getElementById('datasetImportMode').value === 'new') {
      throw new Error('No fully numeric columns were found for the prediction target.');
    }
    const suggestedName = file.name.replace(/\.csv$/i, '').replace(/[_-]+/g, ' ').trim();
    const nameInput = document.getElementById('newDatasetName');
    if (!nameInput.value.trim() || nameInput.dataset.fileDerived === 'true') {
      nameInput.value = suggestedName || 'Imported dataset';
      nameInput.dataset.fileDerived = 'true';
    }
    validateImportedCsv();
  } catch (error) {
    importedCsvRows = null;
    importStatus(error.message, 'error');
    document.getElementById('confirmDatasetImport').disabled = true;
  }
}

function persistNewDataset() {
  const target = document.getElementById('datasetTargetColumn').value;
  const features = window.DatasetUtils.inferFeatures(importedCsvRows, target);
  const label = document.getElementById('newDatasetName').value.trim();
  const id = `custom-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
  const config = {
    id,
    label,
    title: `${label} Prediction`,
    subtitle: `Estimate ${target} from the imported dataset predictors.`,
    target,
    targetLabel: target,
    targetUnit: document.getElementById('newDatasetTargetUnit').value.trim() || 'data units',
    sampleLabel: 'Sample',
    note: `Imported from ${importFileName}; stored locally in this browser.`,
    features,
  };
  const records = storedJson(customDatasetStorageKey, []);
  const record = { config, rows: importedCsvRows };
  localStorage.setItem(customDatasetStorageKey, JSON.stringify([...records, record]));
  window.DATASET_CONFIGS[id] = { ...config, rows: importedCsvRows };
  const selector = document.getElementById('datasetSelect');
  selector.append(datasetOption(config));
  selector.value = id;
  localStorage.setItem('startup-regression:active-dataset', id);
  window.loadDataset(id);
}

function persistAppendedRows() {
  const datasetId = document.getElementById('datasetSelect').value;
  const config = window.DATASET_CONFIGS[datasetId];
  const rows = window.DatasetUtils.validateRows(importedCsvRows, config);
  const storageKey = `${appendedRowsStoragePrefix}${datasetId}`;
  const existing = storedJson(storageKey, []);
  localStorage.setItem(storageKey, JSON.stringify([...existing, ...rows]));
  localStorage.setItem('startup-regression:active-dataset', datasetId);
  window.loadDataset(datasetId);
}

function handleImport() {
  if (!validateImportedCsv()) return;
  try {
    if (document.getElementById('datasetImportMode').value === 'new') {
      persistNewDataset();
    } else {
      persistAppendedRows();
    }
    document.getElementById('datasetDialog').close();
    importedCsvRows = null;
    document.getElementById('datasetFile').value = '';
    document.getElementById('confirmDatasetImport').disabled = true;
  } catch (error) {
    importStatus(`Could not save the imported data: ${error.message}`, 'error');
  }
}

window.DatasetImport = {
  getAppendedRows(datasetId) {
    const rows = storedJson(`${appendedRowsStoragePrefix}${datasetId}`, []);
    return Array.isArray(rows) ? rows : [];
  },
};

window.addEventListener('DOMContentLoaded', () => {
  const dialog = document.getElementById('datasetDialog');
  const fileInput = document.getElementById('datasetFile');
  const nameInput = document.getElementById('newDatasetName');
  const targetSelect = document.getElementById('datasetTargetColumn');
  restoreCustomDatasets();
  setImportMode();

  document.getElementById('datasetSelect').addEventListener('change', (event) => {
    localStorage.setItem('startup-regression:active-dataset', event.target.value);
    validateImportedCsv();
  });
  document.getElementById('openDatasetImport').addEventListener('click', () => {
    importStatus('Choose a CSV file to get started. Imported data stays in this browser.');
    dialog.showModal();
  });
  document.getElementById('closeDatasetDialog').addEventListener('click', () => dialog.close());
  document.getElementById('cancelDatasetImport').addEventListener('click', () => dialog.close());
  document.getElementById('datasetImportMode').addEventListener('change', setImportMode);
  fileInput.addEventListener('change', () => {
    const [file] = fileInput.files;
    if (file) readImportFile(file);
  });
  targetSelect.addEventListener('change', validateImportedCsv);
  document.getElementById('newDatasetTargetUnit').addEventListener('input', validateImportedCsv);
  nameInput.addEventListener('input', () => {
    nameInput.dataset.fileDerived = 'false';
    validateImportedCsv();
  });
  document.getElementById('confirmDatasetImport').addEventListener('click', handleImport);
  dialog.addEventListener('click', (event) => {
    if (event.target === dialog) dialog.close();
  });
});