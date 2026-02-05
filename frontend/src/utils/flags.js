/**
 * Flag utility — resolves team names to flag image URLs via flagcdn.com (free, no API key).
 *
 * flagcdn.com supports:
 *   - Standard ISO 3166-1 alpha-2 codes (lowercase): /w40/us.png
 *   - UK home nations with sub-codes: /w40/gb-eng.png, /w40/gb-sct.png, etc.
 *   - Kosovo (xk) is supported
 *
 * For teams without an ISO code (non-sovereign micro-entities like "Basque Country"),
 * we return null and the UI renders a fallback globe/initials badge.
 */

import teamCodes from '../data/team_codes.json'

const CDN_BASE = 'https://flagcdn.com'

/**
 * Get the ISO alpha-2 code for a team name, or null if unmapped.
 */
export function getTeamCode(teamName) {
  return teamCodes[teamName] || null
}

/**
 * Get a flag image URL for a team name.
 * @param {string} teamName - e.g. "Brazil"
 * @param {number} width - pixel width (flagcdn serves up to 256)
 * @returns {string|null} - URL or null if no mapping
 */
export function getFlagUrl(teamName, width = 40) {
  const code = getTeamCode(teamName)
  if (!code) return null
  return `${CDN_BASE}/w${width}/${code}.png`
}

/**
 * Get a higher-res flag URL for retina displays.
 */
export function getFlagUrl2x(teamName) {
  return getFlagUrl(teamName, 80)
}

/**
 * Get team initials (first 2-3 chars) for fallback display.
 * "United States" -> "US", "Brazil" -> "BRA", "Korea Republic" -> "KOR"
 */
export function getTeamInitials(teamName) {
  if (!teamName) return '??'
  const words = teamName.split(/\s+/)
  if (words.length >= 2) {
    // Multi-word: take first letter of each (max 3)
    return words.slice(0, 3).map(w => w[0]).join('').toUpperCase()
  }
  // Single word: take first 3 chars
  return teamName.slice(0, 3).toUpperCase()
}

/**
 * Check mapping coverage against a list of team names.
 * Logs warnings for unmapped teams in dev mode.
 * @param {string[]} teams - array of team names
 * @returns {{ mapped: number, total: number, unmapped: string[] }}
 */
export function checkCoverage(teams) {
  const unmapped = teams.filter(t => !teamCodes[t])
  const mapped = teams.length - unmapped.length
  const pct = teams.length > 0 ? ((mapped / teams.length) * 100).toFixed(1) : 0

  if (import.meta.env.DEV && unmapped.length > 0) {
    console.warn(
      `[flags] Coverage: ${mapped}/${teams.length} (${pct}%). ` +
      `Unmapped (${unmapped.length}): ${unmapped.join(', ')}`
    )
  }

  return { mapped, total: teams.length, unmapped }
}
