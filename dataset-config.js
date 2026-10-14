window.DATASET_CONFIGS = {
  startups: {
    id: 'startups',
    label: '50 Startups',
    file: './50_Startups.csv',
    title: '50 Startups Profit Prediction',
    subtitle: 'Estimate startup profit from research and development, administration, marketing spend, and state.',
    target: 'Profit',
    targetLabel: 'Profit',
    targetUnit: 'USD',
    sampleLabel: 'Startup',
    features: [
      { key: 'R&D Spend', label: 'R&D Spend', type: 'numeric' },
      { key: 'Administration', label: 'Administration', type: 'numeric' },
      { key: 'Marketing Spend', label: 'Marketing Spend', type: 'numeric' },
      { key: 'State', label: 'State', type: 'categorical', baseline: 'California' },
    ],
  },
  boston: {
    id: 'boston',
    label: 'Boston Housing',
    file: './BostonHousing.csv',
    title: 'Boston Housing Price Prediction',
    subtitle: 'Estimate the median home value from neighborhood, property, accessibility, and demographic characteristics.',
    target: 'medv',
    targetLabel: 'Median home value',
    targetUnit: 'USD thousands',
    sampleLabel: 'District',
    note: 'Historical-data caveat: MEDV is capped at $50,000. B is derived from a racially discriminatory census measure. This dataset is for education, not real housing decisions.',
    features: [
      { key: 'crim', label: 'CRIM · Per-capita crime rate', type: 'numeric' },
      { key: 'zn', label: 'ZN · Residential land zoned', type: 'numeric' },
      { key: 'indus', label: 'INDUS · Non-retail business acres', type: 'numeric' },
      { key: 'chas', label: 'CHAS · Charles River tract', type: 'numeric' },
      { key: 'nox', label: 'NOX · Nitric oxide concentration', type: 'numeric' },
      { key: 'rm', label: 'RM · Average rooms per home', type: 'numeric' },
      { key: 'age', label: 'AGE · Older owner-occupied homes', type: 'numeric' },
      { key: 'dis', label: 'DIS · Distance to employment centers', type: 'numeric' },
      { key: 'rad', label: 'RAD · Highway accessibility index', type: 'numeric' },
      { key: 'tax', label: 'TAX · Property-tax rate', type: 'numeric' },
      { key: 'ptratio', label: 'PTRATIO · Pupil-teacher ratio', type: 'numeric' },
      { key: 'b', label: 'B · Racialized census-derived variable', type: 'numeric' },
      { key: 'lstat', label: 'LSTAT · Lower-status population (%)', type: 'numeric' },
    ],
  },
};

window.DatasetUtils = {
  parseLine(line) {
    const values = [];
    let value = '';
    let quoted = false;
    for (let index = 0; index < line.length; index++) {
      const character = line[index];
      if (character === '"' && quoted && line[index + 1] === '"') {
        value += '"';
        index++;
      } else if (character === '"') {
        quoted = !quoted;
      } else if (character === ',' && !quoted) {
        values.push(value.trim());
        value = '';
      } else {
        value += character;
      }
    }
    values.push(value.trim());
    return values;
  },

  parseCSV(text) {
    const lines = text.replace(/^\uFEFF/, '').trim().split(/\r?\n/);
    if (lines.length < 2 || !lines[0].trim()) throw new Error('The CSV must include a header and at least one data row.');
    const headers = this.parseLine(lines[0]);
    if (headers.some((header) => !header)) throw new Error('CSV column names cannot be blank.');
    if (new Set(headers.map((header) => header.toLowerCase())).size !== headers.length) {
      throw new Error('CSV column names must be unique, ignoring capitalization.');
    }
    return lines.slice(1).filter((line) => line.trim()).map((line, rowIndex) => {
      const values = this.parseLine(line);
      if (values.length !== headers.length) {
        throw new Error(`CSV row ${rowIndex + 2} has ${values.length} values; expected ${headers.length}.`);
      }
      return Object.fromEntries(headers.map((header, index) => {
        const raw = values[index];
        const value = Number(raw);
        return [header, raw !== '' && Number.isFinite(value) ? value : raw];
      }));
    });
  },

  escapeHTML(value) {
    return String(value).replace(/[&<>"']/g, (character) => ({
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#39;',
    })[character]);
  },

  numericColumns(rows) {
    if (!rows.length) return [];
    return Object.keys(rows[0]).filter((key) => rows.every((row) => (
      row[key] !== '' && Number.isFinite(Number(row[key]))
    )));
  },

  inferFeatures(rows, target) {
    if (!rows.length || !Object.hasOwn(rows[0], target)) {
      throw new Error('Select a target column that exists in the CSV.');
    }
    if (!this.numericColumns(rows).includes(target)) {
      throw new Error('The selected prediction target must contain only numeric values.');
    }
    if (new Set(rows.map((row) => Number(row[target]))).size < 2) {
      throw new Error('The selected prediction target must contain more than one distinct value.');
    }
    const numericColumns = this.numericColumns(rows);
    const featureDefinitions = Object.keys(rows[0]).filter((key) => key !== target).flatMap((key) => {
      if (rows.some((row) => row[key] === '')) {
        throw new Error(`Column "${key}" contains blank values. Fill or remove missing values before importing.`);
      }
      if (numericColumns.includes(key)) {
        const uniqueValues = new Set(rows.map((row) => Number(row[key])));
        return uniqueValues.size > 1 ? [{ key, label: key, type: 'numeric' }] : [];
      }
      const levels = [...new Set(rows.map((row) => String(row[key])))];
      if (levels.length === 1) return [];
      if (levels.length > 12) {
        throw new Error(`Column "${key}" has ${levels.length} categories. Encode or remove it before importing (maximum 12).`);
      }
      const sparseLevel = levels.find((level) => rows.filter((row) => String(row[key]) === level).length < 5);
      if (sparseLevel) throw new Error(`Category "${sparseLevel}" in column "${key}" has fewer than 5 rows.`);
      return [{ key, label: key, type: 'categorical', baseline: levels[0] }];
    });
    if (!featureDefinitions.length) throw new Error('The CSV needs at least one predictor column besides the target.');
    const expandedCount = featureDefinitions.reduce((count, feature) => {
      if (feature.type !== 'categorical') return count + 1;
      return count + Math.max(1, new Set(rows.map((row) => String(row[feature.key]))).size - 1);
    }, 0);
    if (expandedCount > 13) {
      throw new Error(`This CSV expands to ${expandedCount} predictors. Keep the model to 13 or fewer numeric/encoded predictors.`);
    }
    if (rows.length - Math.ceil(rows.length * 0.2) <= expandedCount + 2) {
      throw new Error(`This CSV needs more rows for ${expandedCount} predictors. Add at least ${expandedCount + 5} complete records.`);
    }
    return featureDefinitions;
  },

  validateRows(rows, config) {
    if (!rows.length) throw new Error('The CSV has no data rows.');
    const required = [config.target, ...config.features.map((feature) => feature.key)];
    const headers = Object.keys(rows[0]);
    const headerMap = new Map(headers.map((header) => [header.trim().toLowerCase(), header]));
    const mapping = new Map(required.map((key) => [key, headerMap.get(key.trim().toLowerCase())]));
    const missing = required.filter((key) => !mapping.get(key));
    if (missing.length) throw new Error(`Missing required columns: ${missing.join(', ')}.`);

    return rows.map((row, index) => {
      const normalized = {};
      required.forEach((key) => {
        const value = row[mapping.get(key)];
        if (value === '') throw new Error(`Row ${index + 2} is missing a value for "${key}".`);
        const isNumeric = key === config.target
          || config.features.find((feature) => feature.key === key).type === 'numeric';
        if (isNumeric && !Number.isFinite(Number(value))) {
          throw new Error(`Row ${index + 2}, column "${key}" must be numeric.`);
        }
        normalized[key] = isNumeric ? Number(value) : String(value);
      });
      return normalized;
    });
  },

  prepareRows(rawRows, config) {
    const features = [];
    config.features.forEach((definition) => {
      if (definition.type === 'categorical') {
        const levels = [...new Set(rawRows.map((row) => String(row[definition.key])))];
        const requestedBaseline = String(definition.baseline);
        const baseline = levels.includes(requestedBaseline) ? requestedBaseline : levels[0];
        levels.filter((level) => level !== baseline).forEach((level) => {
          features.push({
            key: `${definition.key}=${level}`,
            label: `${definition.label}: ${level}`,
            type: 'dummy',
            source: definition.key,
            level,
          });
        });
      } else {
        features.push({
          key: definition.key,
          label: definition.label,
          type: 'numeric',
          source: definition.key,
        });
      }
    });

    const rows = rawRows.map((raw) => {
      const row = { ...raw, [config.target]: Number(raw[config.target]) };
      features.forEach((feature) => {
        row[feature.key] = feature.type === 'dummy'
          ? Number(String(raw[feature.source]) === String(feature.level))
          : Number(raw[feature.source]);
      });
      return row;
    });
    return { rows, features };
  },

  targetText(value, config, digits = 2) {
    if (config.targetUnit === 'USD') {
      return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: digits,
        maximumFractionDigits: digits,
      }).format(value);
    }
    const formatted = new Intl.NumberFormat('en-US', {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
    }).format(Math.abs(value));
    if (config.targetUnit === 'USD thousands') return `${value < 0 ? '-' : ''}$${formatted}k`;
    const numeric = new Intl.NumberFormat('en-US', {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
    }).format(value);
    return config.targetUnit && config.targetUnit !== 'data units'
      ? `${numeric} ${config.targetUnit}`
      : numeric;
  },
};