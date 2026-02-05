import { useState, useEffect } from 'react'
import PredictionForm from './components/PredictionForm'
import ResultsCard from './components/ResultsCard'
import ModelComparison from './components/ModelComparison'
import { checkCoverage } from './utils/flags'

const API = '/api'

export default function App() {
  const [teams, setTeams] = useState([])
  const [models, setModels] = useState([])
  const [comparison, setComparison] = useState(null)
  const [comparisonMeta, setComparisonMeta] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [apiStatus, setApiStatus] = useState('connecting')
  const [versionInfo, setVersionInfo] = useState(null)

  // Load teams + models on mount
  useEffect(() => {
    async function init() {
      try {
        const [teamsRes, modelsRes, compRes, verRes] = await Promise.all([
          fetch(`${API}/teams`),
          fetch(`${API}/models`),
          fetch(`${API}/comparison`),
          fetch(`${API}/version`),
        ])
        if (!teamsRes.ok) throw new Error('Failed to load teams')
        const teamsData = await teamsRes.json()
        setTeams(teamsData)
        setModels(await modelsRes.json())
        if (compRes.ok) {
          const compData = await compRes.json()
          setComparison(compData.models || compData)
          setComparisonMeta(compData.metadata || null)
        }
        if (verRes.ok) setVersionInfo(await verRes.json())
        setApiStatus('connected')

        // Dev-mode flag coverage check
        checkCoverage(teamsData)
      } catch (err) {
        setApiStatus('error')
        setError('Cannot connect to API. Start the backend: uvicorn wc_predictor.api:app --port 8000')
      }
    }
    init()
  }, [])

  async function handlePredict(formData) {
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const res = await fetch(`${API}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          home_team: formData.homeTeam,
          away_team: formData.awayTeam,
          match_date: formData.date,
          model: formData.model || null,
          neutral: formData.neutral,
        }),
      })
      if (!res.ok) {
        const data = await res.json()
        throw new Error(data.detail || 'Prediction failed')
      }
      setResult(await res.json())
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header className="header">
        <h1 className="header__logo">
          Prometheus
          <span>World Cup Intelligence</span>
        </h1>
        <p className="header__sub">
          ML-powered match outcome prediction for FIFA World Cup
        </p>
        <div style={{ marginTop: '0.8rem' }}>
          <span className={`status ${apiStatus === 'error' ? 'status--error' : ''}`}>
            <span className="status__dot" />
            {apiStatus === 'connected' ? `${teams.length} teams | ${models.length} models loaded` :
             apiStatus === 'error' ? 'API offline' : 'Connecting...'}
          </span>
        </div>
      </header>

      <div className="grid grid--2" style={{ marginBottom: '1.5rem' }}>
        <PredictionForm
          teams={teams}
          models={models}
          loading={loading}
          onPredict={handlePredict}
        />
        <ResultsCard result={result} loading={loading} error={error} />
      </div>

      <ModelComparison comparison={comparison} metadata={comparisonMeta} models={models} />

      <footer className="footer">
        Prometheus {versionInfo ? `v${versionInfo.api_version}` : 'v1.1'} &mdash;
        ELO + ML prediction pipeline
        {versionInfo?.git_commit && (
          <span> &mdash; {versionInfo.git_commit}</span>
        )}
        {versionInfo?.dataset_hash && (
          <span> &mdash; data:{versionInfo.dataset_hash}</span>
        )}
        <br />
        Built with FastAPI + React &mdash; Flags via flagcdn.com
      </footer>
    </div>
  )
}
