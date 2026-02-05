import { useState } from 'react'

export default function ModelComparison({ comparison, metadata, models }) {
  const [sortBy, setSortBy] = useState('log_loss')
  const [sortAsc, setSortAsc] = useState(true)

  if (!comparison) return null

  const metrics = ['accuracy', 'macro_f1', 'log_loss', 'brier_score']
  const metricLabels = {
    accuracy: 'Accuracy',
    macro_f1: 'Macro F1',
    log_loss: 'Log Loss',
    brier_score: 'Brier Score',
  }

  // Sort model names
  const modelNames = Object.keys(comparison).sort((a, b) => {
    const va = comparison[a]?.[sortBy] ?? Infinity
    const vb = comparison[b]?.[sortBy] ?? Infinity
    return sortAsc ? va - vb : vb - va
  })

  // For each metric, find the best (highest for accuracy/f1, lowest for loss/brier)
  function isBest(metric, modelName) {
    const nonBaseline = Object.keys(comparison).filter(n => n !== 'baseline')
    const values = nonBaseline.map(n => comparison[n]?.[metric]).filter(v => v != null)
    const val = comparison[modelName]?.[metric]
    if (val == null) return false
    if (metric === 'log_loss' || metric === 'brier_score') {
      return val === Math.min(...values)
    }
    return val === Math.max(...values)
  }

  function handleSort(metric) {
    if (sortBy === metric) {
      setSortAsc(!sortAsc)
    } else {
      setSortBy(metric)
      // Default: lower-is-better metrics sort ascending, higher-is-better descending
      setSortAsc(metric === 'log_loss' || metric === 'brier_score')
    }
  }

  return (
    <div className="card" style={{ marginTop: '0' }}>
      <h2 className="card__title">Model Comparison</h2>

      {/* Metadata banner */}
      {metadata && (
        <div className="comparison-meta">
          <span>Test period: post-{metadata.cutoff_year}</span>
          {metadata.test_matches > 0 && (
            <span> &middot; {metadata.test_matches} test matches</span>
          )}
        </div>
      )}

      <div style={{ overflowX: 'auto' }}>
        <table className="model-table">
          <thead>
            <tr>
              <th>Model</th>
              {metrics.map(m => (
                <th
                  key={m}
                  onClick={() => handleSort(m)}
                  style={{ cursor: 'pointer', userSelect: 'none' }}
                  title={`Sort by ${metricLabels[m]}`}
                >
                  {metricLabels[m]}
                  {sortBy === m && (
                    <span style={{ marginLeft: '4px', color: 'var(--accent)' }}>
                      {sortAsc ? '\u25B2' : '\u25BC'}
                    </span>
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {modelNames.map(name => (
              <tr key={name} className={name === 'baseline' ? 'model-table__baseline' : ''}>
                <td style={{ color: 'var(--text-primary)', fontWeight: 500 }}>
                  {name}
                  {name === 'baseline' && (
                    <span className="baseline-tag" title="DummyClassifier reference point">ref</span>
                  )}
                </td>
                {metrics.map(m => {
                  const val = comparison[name]?.[m]
                  const best = isBest(m, name) && name !== 'baseline'
                  return (
                    <td key={m} className={best ? 'best' : ''}>
                      {val != null ? val.toFixed(4) : '\u2014'}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Baseline explanation note */}
      {metadata?.note && (
        <div className="comparison-note">
          <strong>Note:</strong> {metadata.note}
        </div>
      )}
    </div>
  )
}
