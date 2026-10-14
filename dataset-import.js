const customDatasetStorageKey = 'startup-regression:custom-datasets';
const appendedRowsStoragePrefix = 'startup-regression:appended:';

let importedCsvRows = null;
let importFileName = '';
let importSourceDescription = '';

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

function datasetNameKey(name) {
  return String(name).trim().replace(/\s+/g, ' ').toLocaleLowerCase();
}

function uniqueCustomRecords(records) {
  if (!Array.isArray(records)) return [];
  const byName = new Map();
  const builtInNames = new Set(Object.values(window.DATASET_CONFIGS)
    .filter((config) => config.file)
    .map((config) => datasetNameKey(config.label)));
  records.forEach((record) => {
    if (!record?.config?.id || !record.config.label || !Array.isArray(record.rows)) return;
    const key = datasetNameKey(record.config.label);
    if (!key || builtInNames.has(key)) return;
    byName.delete(key);
    byName.set(key, record);
  });
  return [...byName.values()];
}

function restoreCustomDatasets() {
  const storedRecords = storedJson(customDatasetStorageKey, []);
  const records = uniqueCustomRecords(Array.isArray(storedRecords) ? storedRecords : []);
  if (records.length !== storedRecords.length) {
    localStorage.setItem(customDatasetStorageKey, JSON.stringify(records));
  }
  const selector = document.getElementById('datasetSelect');
  records.forEach((record) => {
    if (!record?.config?.id || !Array.isArray(record.rows)) return;
    const config = { ...record.config, rows: record.rows };
    window.DATASET_CONFIGS[config.id] = config;
    const duplicateOption = [...selector.options].find((option) => option.value === config.id);
    if (duplicateOption) {
      duplicateOption.textContent = config.label;
    } else {
      selector.append(datasetOption(config));
    }
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

function setImportSource() {
  const source = document.getElementById('datasetSource').value;
  const isUrl = source === 'url';
  document.getElementById('datasetFileField').hidden = isUrl;
  document.getElementById('datasetUrlField').hidden = !isUrl;
  importedCsvRows = null;
  document.getElementById('confirmDatasetImport').disabled = true;
  validateUrlInput();
  if (!isUrl) importStatus('Choose a CSV file to load it.');
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

async function processCsvText(text, sourceName, sourceDescription) {
  importedCsvRows = null;
  importFileName = sourceName;
  importSourceDescription = sourceDescription;
  document.getElementById('confirmDatasetImport').disabled = true;
  importStatus('Reading CSV…');
  try {
    if (new Blob([text]).size > 1_500_000) throw new Error('CSV files must be 1.5 MB or smaller for browser-local storage.');
    const rows = window.DatasetUtils.parseCSV(text);
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
    const suggestedName = sourceName.replace(/\.csv$/i, '').replace(/[_-]+/g, ' ').trim();
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

async function readImportFile(file) {
  try {
    if (file.size > 1_500_000) throw new Error('CSV files must be 1.5 MB or smaller for browser-local storage.');
    await processCsvText(await file.text(), file.name, `file: ${file.name}`);
  } catch (error) {
    importedCsvRows = null;
    importStatus(error.message, 'error');
    document.getElementById('confirmDatasetImport').disabled = true;
  }
}

function validateUrlInput() {
  const urlInput = document.getElementById('datasetUrl');
  const loadButton = document.getElementById('loadDatasetUrl');
  let valid = false;
  try {
    const url = new URL(urlInput.value.trim());
    valid = ['http:', 'https:'].includes(url.protocol)
      && !url.username
      && !url.password
      && url.href.length <= 2048;
  } catch {
    valid = false;
  }
  loadButton.disabled = !valid;
  return valid;
}

async function readImportUrl() {
  if (!validateUrlInput()) {
    importStatus('Enter a valid public HTTP or HTTPS CSV URL.', 'error');
    return;
  }
  const url = new URL(document.getElementById('datasetUrl').value.trim());
  document.getElementById('loadDatasetUrl').disabled = true;
  importedCsvRows = null;
  document.getElementById('confirmDatasetImport').disabled = true;
  importStatus('Fetching CSV URL…');
  try {
    const response = await fetch(url.href, { headers: { Accept: 'text/csv,text/plain,*/*' } });
    if (!response.ok) throw new Error(`CSV request failed (${response.status} ${response.statusText}).`);
    const text = await response.text();
    if (new Blob([text]).size > 1_500_000) {
      throw new Error('CSV responses must be 1.5 MB or smaller for browser-local storage.');
    }
    const lastPathPart = decodeURIComponent(url.pathname.split('/').filter(Boolean).pop() || 'remote-dataset.csv');
    const sourceName = lastPathPart.toLowerCase().endsWith('.csv') ? lastPathPart : `${lastPathPart}.csv`;
    await processCsvText(text, sourceName, url.href);
  } catch (error) {
    importedCsvRows = null;
    const isCorsFailure = error instanceof TypeError;
    importStatus(isCorsFailure
      ? 'Could not read this URL. The server may block browser CORS requests; use a GitHub Raw URL or upload the CSV file instead.'
      : error.message, 'error');
    document.getElementById('confirmDatasetImport').disabled = true;
  } finally {
    validateUrlInput();
  }
}

function persistNewDataset() {
  const target = document.getElementById('datasetTargetColumn').value;
  const features = window.DatasetUtils.inferFeatures(importedCsvRows, target);
  const label = document.getElementById('newDatasetName').value.trim();
  const records = uniqueCustomRecords(storedJson(customDatasetStorageKey, []));
  const existingIndex = records.findIndex((record) => datasetNameKey(record.config.label) === datasetNameKey(label));
  const existingRecord = existingIndex >= 0 ? records[existingIndex] : null;
  const id = existingRecord?.config.id
    || `custom-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
  const config = {
    id,
    label,
    title: `${label} Prediction`,
    subtitle: `Estimate ${target} from the imported dataset predictors.`,
    target,
    targetLabel: target,
    targetUnit: document.getElementById('newDatasetTargetUnit').value.trim() || 'data units',
    sampleLabel: 'Sample',
    note: `Imported from ${importSourceDescription || importFileName}; stored locally in this browser.`,
    features,
  };
  const record = { config, rows: importedCsvRows };
  if (existingIndex >= 0) {
    records[existingIndex] = record;
    importStatus(`Updated existing dataset "${label}" with ${importedCsvRows.length} rows.`, 'success');
  } else {
    records.push(record);
  }
  localStorage.setItem(customDatasetStorageKey, JSON.stringify(records));
  window.DATASET_CONFIGS[id] = { ...config, rows: importedCsvRows };
  const selector = document.getElementById('datasetSelect');
  [...selector.options].forEach((option) => {
    if (option.value !== id && datasetNameKey(option.textContent) === datasetNameKey(label)) {
      option.remove();
    }
  });
  const existingOption = [...selector.options].find((option) => option.value === id);
  if (existingOption) {
    existingOption.textContent = label;
  } else {
    selector.append(datasetOption(config));
  }
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
    document.getElementById('datasetUrl').value = '';
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
  const urlInput = document.getElementById('datasetUrl');
  const nameInput = document.getElementById('newDatasetName');
  const targetSelect = document.getElementById('datasetTargetColumn');
  restoreCustomDatasets();
  setImportMode();
  setImportSource();

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
  document.getElementById('datasetSource').addEventListener('change', setImportSource);
  fileInput.addEventListener('change', () => {
    const [file] = fileInput.files;
    if (file) readImportFile(file);
  });
  urlInput.addEventListener('input', validateUrlInput);
  document.getElementById('loadDatasetUrl').addEventListener('click', readImportUrl);
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