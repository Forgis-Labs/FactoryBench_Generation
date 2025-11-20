import type { LinksFunction, MetaFunction } from "@remix-run/node";
import {
  Links,
  LiveReload,
  Meta,
  Outlet,
  Scripts,
  ScrollRestoration,
  useLocation,
} from "@remix-run/react";
import globalStylesUrl from "./styles/global.css";

export const links: LinksFunction = () => [
  { rel: "stylesheet", href: globalStylesUrl },
  { rel: "icon", type: "image/x-icon", href: "/favicon_io/favicon.ico" },
  { rel: "icon", type: "image/png", sizes: "16x16", href: "/favicon_io/favicon-16x16.png" },
  { rel: "icon", type: "image/png", sizes: "32x32", href: "/favicon_io/favicon-32x32.png" },
  { rel: "apple-touch-icon", sizes: "180x180", href: "/favicon_io/apple-touch-icon.png" },
  { rel: "manifest", href: "/favicon_io/site.webmanifest" },
];

export const meta: MetaFunction = () => ([
  { title: "FactoryBench" },
  { name: "viewport", content: "width=device-width, initial-scale=1" },
]);

export default function App() {
  const location = useLocation();
  const path = location.pathname;
  const isActive = (href: string) => (path === href || (href !== '/' && path.startsWith(href))) ? 'active' : '';
  return (
    <html lang="en">
      <head>
        <Meta />
        <Links />
      </head>
      <body>
        <header className="header">
          <div className="container">
            <a className="brand" href="/">
              <img src="/Forgis_white logomark.png" alt="Forgis" style={{ height: 32, width: 32, objectFit: "contain", marginRight: 8 }} />
              FactoryBench
            </a>
            <nav className="nav">
              <a className={isActive('/leaderboard')} href="/leaderboard">Leaderboard</a>
              <a className={isActive('/analysis')} href="/analysis">Analysis</a>
              <a className={isActive('/run')} href="/run">Run</a>
              <a className={isActive('/readme')} href="/readme">Readme</a>
            </nav>
          </div>
        </header>
        <main className="container" style={{ marginTop: 32 }}>
          <Outlet />
        </main>
        <footer className="footer">
          <div className="container">Forgis • telemetry_literacy MVP</div>
        </footer>
        <ScrollRestoration />
        <Scripts />
        <LiveReload />
      </body>
    </html>
  );
}
