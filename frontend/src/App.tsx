import { BookOpen, ClipboardList, Home, Moon, Sun } from "lucide-react";
import { useEffect, useState } from "react";
import { NavLink, Route, Routes } from "react-router-dom";
import HomePage from "./pages/HomePage";
import ProblemDetailPage from "./pages/ProblemDetailPage";
import ProblemsPage from "./pages/ProblemsPage";
import SubmissionsPage from "./pages/SubmissionsPage";
import { cx } from "./utils";

function ThemeButton() {
  const [dark, setDark] = useState(() => localStorage.getItem("dlcode-theme") === "dark");

  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
    localStorage.setItem("dlcode-theme", dark ? "dark" : "light");
  }, [dark]);

  return (
    <button
      type="button"
      onClick={() => setDark((value) => !value)}
      className="inline-flex h-9 w-9 items-center justify-center rounded-md border border-zinc-300 bg-white text-zinc-700 hover:border-moss hover:text-moss dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-200"
      title={dark ? "切换浅色模式" : "切换深色模式"}
    >
      {dark ? <Sun size={18} /> : <Moon size={18} />}
    </button>
  );
}

function NavItem({ to, icon: Icon, label }: { to: string; icon: typeof Home; label: string }) {
  return (
    <NavLink
      to={to}
      end={to === "/"}
      className={({ isActive }) =>
        cx(
          "inline-flex items-center gap-2 rounded-md px-3 py-2 text-sm font-medium transition",
          isActive
            ? "bg-moss text-white"
            : "text-zinc-700 hover:bg-zinc-100 dark:text-zinc-200 dark:hover:bg-zinc-800",
        )
      }
    >
      <Icon size={17} />
      {label}
    </NavLink>
  );
}

export default function App() {
  return (
    <div className="min-h-screen bg-paper text-ink dark:bg-ink dark:text-zinc-100">
      <header className="sticky top-0 z-30 border-b border-zinc-200 bg-paper/95 backdrop-blur dark:border-zinc-800 dark:bg-ink/95">
        <div className="mx-auto flex h-14 max-w-[1500px] items-center justify-between px-4">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-md bg-moss text-lg font-bold text-white">
              DL
            </div>
            <div>
              <div className="text-sm font-semibold leading-4">DLCode</div>
              <div className="text-xs text-zinc-500 dark:text-zinc-400">深度学习手撕题库</div>
            </div>
          </div>
          <nav className="hidden items-center gap-1 md:flex">
            <NavItem to="/" icon={Home} label="首页" />
            <NavItem to="/problems" icon={BookOpen} label="题库" />
            <NavItem to="/submissions" icon={ClipboardList} label="提交记录" />
          </nav>
          <ThemeButton />
        </div>
        <nav className="flex gap-1 border-t border-zinc-200 px-3 py-2 md:hidden dark:border-zinc-800">
          <NavItem to="/" icon={Home} label="首页" />
          <NavItem to="/problems" icon={BookOpen} label="题库" />
          <NavItem to="/submissions" icon={ClipboardList} label="提交记录" />
        </nav>
      </header>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/problems" element={<ProblemsPage />} />
        <Route path="/problems/:slug" element={<ProblemDetailPage />} />
        <Route path="/submissions" element={<SubmissionsPage />} />
      </Routes>
    </div>
  );
}
