import { useState } from 'react'
import TeamFlag from './TeamFlag'

export default function ResultsCard({ result, loading, error }) {
  const [showExplanation, setShowExplanation] = useState(false)

  if (error) {
    return (
      <div className="card">
        <h2 className="card__title">Prediction Output</h2>
        <div className="error-msg">{error}</div>
      </div>
    )
  }

  if (!result && !loading) {
    return (
      <div className="card">
        <h2 className="card__title">Prediction Output</h2>
        <div style={{ textAlign: 'center', padding: '2rem 0' }}>
          <p style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', fontSize: '0.8rem' }}>
            Configure a match and run prediction
          </p>
        </div>
      </div>
    )
  }

  if (loading) {
    return (
      <div className="card">
        <h2 className="card__title">Prediction Output</h2>
        <div style={{ textAlign: 'center', padding: '2rem 0' }}>
          <div className="status">
            <span className="status__dot" />
            Computing ELO ratings and features...
          </div>
        </div>
      </div>
    )
  }

  const probs = result.probabilities
  const conf = result.confidence
  const explanation = result.explanation

  return (
    <div className="card card--accent fade-in">
      <h2 className="card__title">Prediction Output</h2>

      {/* Match header with flags */}
      <div className="match-header">
        <div className="match-header__team">
          <TeamFlag team={result.home_team} size={36} />
          <span className="match-header__name">{result.home_team}</span>
          <span className="match-header__role">HOME</span>
        </div>
        <div className="match-header__center">
          <span className="match-header__vs">VS</span>
          <span className="match-header__date">{result.match_date}</span>
          {explanation?.neutral_venue && (
            <span className="match-header__venue">Neutral</span>
          )}
        </div>
        <div className="match-header__team">
          <TeamFlag team={result.away_team} size={36} />
          <span className="match-header__name">{result.away_team}</span>
          <span className="match-header__role">AWAY</span>
        </div>
      </div>

      <ProbBar label="Home Win" value={probs.H} type="h" />
      <ProbBar label="Draw" value={probs.D} type="d" />
      <ProbBar label="Away Win" value={probs.A} type="a" />

      <div className="verdict">
        <div className="verdict__label">Predicted Outcome</div>
        <div className="verdict__result">{result.prediction_label}</div>
        {conf && (
          <div style={{ marginTop: '0.5rem' }}>
            <span className={`confidence-badge confidence-badge--${conf.label.toLowerCase()}`}>
              {conf.label} CONFIDENCE
            </span>
            {/* Confidence gauge meter */}
            <div className="confidence-gauge">
              <div className="confidence-gauge__track">
                <div
                  className={`confidence-gauge__fill confidence-gauge__fill--${conf.label.toLowerCase()}`}
                  style={{ width: `${conf.max_prob * 100}%` }}
                />
                <div
                  className="confidence-gauge__marker"
                  style={{ left: `${conf.max_prob * 100}%` }}
                />
              </div>
              <div className="confidence-gauge__labels">
                <span>0%</span>
                <span>50%</span>
                <span>100%</span>
              </div>
            </div>
            <div className="verdict__confidence-detail">
              {(conf.max_prob * 100).toFixed(1)}% max prob &middot; entropy {conf.entropy.toFixed(3)}
            </div>
          </div>
        )}
        <div className="verdict__sub">
          Model: {result.model} &middot; {result.match_date}
        </div>
      </div>

      {/* Why This Prediction? */}
      {explanation && (
        <div className="explain-section">
          <button
            className="explain-toggle"
            onClick={() => setShowExplanation(!showExplanation)}
          >
            {showExplanation ? '- Hide' : '+ Why'} this prediction?
          </button>
          {showExplanation && (
            <div className="explain-card fade-in">
              {/* ELO Ratings */}
              {explanation.elo && (
                <div className="explain-block">
                  <div className="explain-block__title">ELO Ratings</div>
                  <div className="explain-grid">
                    <ExplainStat label="Home ELO" value={explanation.elo.home_elo} />
                    <ExplainStat label="Away ELO" value={explanation.elo.away_elo} />
                    <ExplainStat
                      label="ELO Diff"
                      value={explanation.elo.elo_diff > 0 ? `+${explanation.elo.elo_diff}` : explanation.elo.elo_diff}
                      accent={explanation.elo.elo_diff > 0}
                    />
                    <ExplainStat label="ELO P(H)" value={`${(explanation.elo.elo_home_win_prob * 100).toFixed(1)}%`} />
                    <ExplainStat label="ELO P(D)" value={`${(explanation.elo.elo_draw_prob * 100).toFixed(1)}%`} />
                    <ExplainStat label="ELO P(A)" value={`${(explanation.elo.elo_away_win_prob * 100).toFixed(1)}%`} />
                  </div>
                </div>
              )}

              {/* Recent Form */}
              <div className="explain-block">
                <div className="explain-block__title">Recent Form (Last 5)</div>
                <div className="explain-grid explain-grid--2">
                  <FormSummary team={result.home_team} form={explanation.home_form_last5} />
                  <FormSummary team={result.away_team} form={explanation.away_form_last5} />
                </div>
              </div>

              {/* Head-to-Head */}
              {explanation.head_to_head && (
                <div className="explain-block">
                  <div className="explain-block__title">
                    Head-to-Head (Last {explanation.head_to_head.last_n})
                  </div>
                  <div className="explain-grid">
                    <ExplainStat label={`${result.home_team} Wins`} value={explanation.head_to_head.home_wins} />
                    <ExplainStat label="Draws" value={explanation.head_to_head.draws} />
                    <ExplainStat label={`${result.away_team} Wins`} value={explanation.head_to_head.away_wins} />
                  </div>
                </div>
              )}
              {!explanation.head_to_head && (
                <div className="explain-block">
                  <div className="explain-block__title">Head-to-Head</div>
                  <div style={{ color: 'var(--text-muted)', fontSize: '0.75rem', fontFamily: 'var(--font-mono)' }}>
                    No prior meetings found
                  </div>
                </div>
              )}

              {/* Context */}
              <div className="explain-block">
                <div className="explain-block__title">Context</div>
                <div className="explain-grid">
                  <ExplainStat label="Venue" value={explanation.neutral_venue ? 'Neutral' : 'Home Advantage'} />
                  <ExplainStat label="Home Confed." value={explanation.home_confederation} />
                  <ExplainStat label="Away Confed." value={explanation.away_confederation} />
                  <ExplainStat label="History Size" value={`${explanation.total_matches_in_history?.toLocaleString()} matches`} />
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function ProbBar({ label, value, type }) {
  const pct = (value * 100).toFixed(1)
  return (
    <div className="prob-row">
      <span className="prob-row__label">{label}</span>
      <div className="prob-row__bar-bg">
        <div
          className={`prob-row__bar prob-row__bar--${type}`}
          style={{ width: `${Math.max(pct, 3)}%` }}
        >
          <span className="prob-row__value">{pct}%</span>
        </div>
      </div>
    </div>
  )
}

function ExplainStat({ label, value, accent }) {
  return (
    <div className="explain-stat">
      <div className="explain-stat__label">{label}</div>
      <div className={`explain-stat__value ${accent ? 'explain-stat__value--accent' : ''}`}>
        {value}
      </div>
    </div>
  )
}

function FormSummary({ team, form }) {
  if (!form || form.matches === 0) {
    return (
      <div className="form-summary">
        <div className="form-summary__team">
          <TeamFlag team={team} size={16} />
          <span>{team}</span>
        </div>
        <div style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>No recent data</div>
      </div>
    )
  }
  return (
    <div className="form-summary">
      <div className="form-summary__team">
        <TeamFlag team={team} size={16} />
        <span>{team}</span>
      </div>
      <div className="form-summary__record">
        <span className="form-w">{form.W}W</span>
        <span className="form-d">{form.D}D</span>
        <span className="form-l">{form.L}L</span>
      </div>
      <div className="form-summary__goals">
        {form.goals_for_avg != null && (
          <span>GF {form.goals_for_avg} / GA {form.goals_against_avg}</span>
        )}
      </div>
    </div>
  )
}
