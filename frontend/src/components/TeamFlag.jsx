import { useState } from 'react'
import { getFlagUrl, getFlagUrl2x, getTeamInitials } from '../utils/flags'

/**
 * Renders a flag image for a team, with graceful fallback to initials badge.
 *
 * @param {string} team - Team display name (e.g. "Brazil")
 * @param {number} size - Display size in px (default 24)
 * @param {string} className - Optional extra CSS class
 */
export default function TeamFlag({ team, size = 24, className = '' }) {
  const [imgError, setImgError] = useState(false)
  const flagUrl = getFlagUrl(team)
  const flagUrl2x = getFlagUrl2x(team)

  if (!flagUrl || imgError) {
    // Fallback: initials badge
    return (
      <span
        className={`team-flag team-flag--fallback ${className}`}
        style={{
          width: size,
          height: size,
          fontSize: size * 0.35,
          lineHeight: `${size}px`,
        }}
        title={team}
      >
        {getTeamInitials(team)}
      </span>
    )
  }

  return (
    <img
      className={`team-flag ${className}`}
      src={flagUrl}
      srcSet={`${flagUrl} 1x, ${flagUrl2x} 2x`}
      alt={`${team} flag`}
      title={team}
      width={size}
      height={Math.round(size * 0.75)} // flags are ~4:3 aspect
      onError={() => setImgError(true)}
      loading="lazy"
    />
  )
}
