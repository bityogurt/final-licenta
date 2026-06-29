import { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, Navigate, useLocation } from 'react-router-dom';
import MarketDiscovery from './components/MarketDiscovery';
import Login from './components/Login';
import CreateOrder from './components/CreateOrder';
import Inventory from './components/Inventory';

// Componenta pentru a gestiona aspectul tab-ului activ
function NavTab({ to, children }) {
    const location = useLocation();
    const isActive = location.pathname === to;
    
    return (
        <Link to={to} style={{
            color: isActive ? '#ffffff' : '#9ca3af',
            textDecoration: 'none',
            padding: '8px 16px',
            borderRadius: '6px',
            backgroundColor: isActive ? '#374151' : 'transparent',
            fontWeight: isActive ? 'bold' : 'normal',
            transition: 'all 0.2s ease'
        }}>
            {children}
        </Link>
    );
}

export default function App() {
    const [isAuthenticated, setIsAuthenticated] = useState(() => {
        return localStorage.getItem('isAuthenticated') === 'true';
    });

    const handleLogout = () => {
        localStorage.removeItem('isAuthenticated');
        setIsAuthenticated(false);
    };

    return (
        <Router>
            <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', backgroundColor: '#f9fafb', fontFamily: 'system-ui, -apple-system, sans-serif' }}>
                <header style={{ 
                    backgroundColor: '#111827', 
                    padding: '16px 32px', 
                    color: 'white', 
                    display: 'flex', 
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '40px' }}>
                        <h1 style={{ margin: 0, fontSize: '1.25rem', letterSpacing: '0.05em' }}>ADMIN SISTEM</h1>
                        
                        {isAuthenticated && (
                            <nav style={{ display: 'flex', gap: '10px' }}>
                                <NavTab to="/inventory">Inventar</NavTab>
                                <NavTab to="/orders">Vanzari</NavTab>
                                <NavTab to="/discovery">Market</NavTab>
                            </nav>
                        )}
                    </div>
                    
                    {isAuthenticated && (
                        <button 
                            onClick={handleLogout} 
                            style={{ 
                                background: '#ef4444', 
                                border: 'none', 
                                color: 'white', 
                                padding: '8px 16px',
                                borderRadius: '6px',
                                cursor: 'pointer', 
                                fontWeight: 'bold',
                                transition: 'background-color 0.2s'
                            }}
                        >
                            Deconectare
                        </button>
                    )}
                </header>
                
                <main style={{ flex: 1, padding: '32px' }}>
                    <Routes>
                        <Route path="/login" element={isAuthenticated ? <Navigate to="/inventory" /> : <Login setAuth={setIsAuthenticated} />} />
                        <Route path="/inventory" element={isAuthenticated ? <Inventory /> : <Navigate to="/login" />} />
                        <Route path="/orders" element={isAuthenticated ? <CreateOrder /> : <Navigate to="/login" />} />
                        <Route path="/discovery" element={isAuthenticated ? <MarketDiscovery /> : <Navigate to="/login" />} />
                        <Route path="*" element={<Navigate to="/login" />} />
                    </Routes>
                </main>
            </div>
        </Router>
    );
}