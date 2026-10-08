import { useEffect, useState } from 'react'

export type Route = { page: string; params: URLSearchParams }

function parse(): Route {
  const [page, query = ''] = window.location.hash.slice(1).split('?')
  return { page: page || 'overview', params: new URLSearchParams(query) }
}

/** Hash routes like `#night?rec=SC4071&epoch=1145`, so every view can be linked to. */
export function useRoute() {
  const [route, setRoute] = useState<Route>(parse)
  useEffect(() => {
    const onHash = () => setRoute(parse())
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])
  return route
}

export function navigate(page: string, params: Record<string, string | number> = {}) {
  const query = new URLSearchParams(Object.entries(params).map(([k, v]) => [k, String(v)])).toString()
  window.location.assign(`#${page}${query ? `?${query}` : ''}`)
  window.scrollTo({ top: 0 })
}

/** Update the current URL's parameters without adding a history entry. */
export function replaceParams(page: string, params: Record<string, string | number>) {
  const query = new URLSearchParams(Object.entries(params).map(([k, v]) => [k, String(v)])).toString()
  window.history.replaceState(null, '', `#${page}${query ? `?${query}` : ''}`)
}
