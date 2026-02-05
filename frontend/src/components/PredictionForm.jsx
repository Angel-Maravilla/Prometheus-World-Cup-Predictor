import { useState, useRef, useEffect } from 'react'
import TeamFlag from './TeamFlag'

export default function PredictionForm({ teams, models, loading, onPredict }) {
  const [homeTeam, setHomeTeam] = useState('')
  const [awayTeam, setAwayTeam] = useState('')
  const [date, setDate] = useState('2026-06-15')
  const [model, setModel] = useState('')
  const [neutral, setNeutral] = useState(true)

  function handleSubmit(e) {
    e.preventDefault()
    if (!homeTeam || !awayTeam) return
    onPredict({ homeTeam, awayTeam, date, model, neutral })
  }

  function handleSwap() {
    const prevHome = homeTeam
    setHomeTeam(awayTeam)
    setAwayTeam(prevHome)
  }

  return (
    <div className="card">
      <h2 className="card__title">Match Configuration</h2>
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label>Home Team</label>
          <SearchableSelect
            options={teams}
            value={homeTeam}
            onChange={setHomeTeam}
            disabledOption={awayTeam}
            placeholder="Search teams..."
          />
        </div>

        {/* Swap button */}
        <div style={{ display: 'flex', justifyContent: 'center', margin: '-0.3rem 0 0.5rem' }}>
          <button
            type="button"
            className="swap-btn"
            onClick={handleSwap}
            disabled={!homeTeam && !awayTeam}
            title="Swap home and away teams"
          >
            &#x21C5; Swap
          </button>
        </div>

        <div className="form-group">
          <label>Away Team</label>
          <SearchableSelect
            options={teams}
            value={awayTeam}
            onChange={setAwayTeam}
            disabledOption={homeTeam}
            placeholder="Search teams..."
          />
        </div>

        <div className="form-group">
          <label>Match Date</label>
          <input
            type="date"
            value={date}
            onChange={e => setDate(e.target.value)}
            required
          />
        </div>

        <div className="form-group">
          <label>Model</label>
          <div className="model-pills">
            <button
              type="button"
              className={`model-pill ${model === '' ? 'model-pill--active' : ''}`}
              onClick={() => setModel('')}
            >
              Auto
            </button>
            {models.map(m => (
              <button
                type="button"
                key={m.name}
                className={`model-pill ${model === m.name ? 'model-pill--active' : ''}`}
                onClick={() => setModel(m.name)}
              >
                {m.name}
              </button>
            ))}
          </div>
        </div>

        <div className="form-group" style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
          <input
            type="checkbox"
            id="neutral"
            checked={neutral}
            onChange={e => setNeutral(e.target.checked)}
            style={{ width: 'auto', accentColor: 'var(--accent)' }}
          />
          <label htmlFor="neutral" style={{ marginBottom: 0 }}>Neutral venue</label>
        </div>

        <button
          type="submit"
          className={`btn ${loading ? 'btn--loading' : ''}`}
          disabled={loading || !homeTeam || !awayTeam || homeTeam === awayTeam}
        >
          {loading ? 'Analyzing' : 'Run Prediction'}
        </button>
      </form>
    </div>
  )
}

/**
 * Searchable dropdown with flag icons.
 */
function SearchableSelect({ options, value, onChange, disabledOption, placeholder }) {
  const [query, setQuery] = useState('')
  const [open, setOpen] = useState(false)
  const wrapperRef = useRef(null)

  useEffect(() => {
    function handleClickOutside(e) {
      if (wrapperRef.current && !wrapperRef.current.contains(e.target)) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  const filtered = options.filter(t =>
    t.toLowerCase().includes(query.toLowerCase()) && t !== disabledOption
  )

  function handleSelect(team) {
    onChange(team)
    setQuery('')
    setOpen(false)
  }

  function handleInputChange(e) {
    setQuery(e.target.value)
    setOpen(true)
    if (e.target.value === '') {
      onChange('')
    }
  }

  function handleFocus() {
    setOpen(true)
    setQuery('')
  }

  return (
    <div className="searchable-select" ref={wrapperRef}>
      {/* Selected value display with flag */}
      {value && !open && (
        <div className="searchable-select__selected" onClick={handleFocus}>
          <TeamFlag team={value} size={20} />
          <span className="searchable-select__selected-name">{value}</span>
          <button
            type="button"
            className="searchable-select__clear"
            onClick={(e) => { e.stopPropagation(); onChange(''); setQuery(''); }}
            title="Clear selection"
          >
            &times;
          </button>
        </div>
      )}
      {/* Search input (visible when open or no selection) */}
      {(!value || open) && (
        <input
          type="text"
          className="searchable-select__input"
          value={query}
          onChange={handleInputChange}
          onFocus={handleFocus}
          placeholder={placeholder}
          autoFocus={open && !!value}
        />
      )}
      {open && (
        <div className="searchable-select__dropdown">
          {filtered.length === 0 && (
            <div className="searchable-select__empty">No matches</div>
          )}
          {filtered.slice(0, 50).map(t => (
            <div
              key={t}
              className={`searchable-select__option ${t === value ? 'searchable-select__option--active' : ''}`}
              onMouseDown={() => handleSelect(t)}
            >
              <TeamFlag team={t} size={18} />
              <span>{t}</span>
            </div>
          ))}
          {filtered.length > 50 && (
            <div className="searchable-select__empty">
              Type to narrow ({filtered.length} results)
            </div>
          )}
        </div>
      )}
    </div>
  )
}
