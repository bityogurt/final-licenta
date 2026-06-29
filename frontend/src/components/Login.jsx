import { useState } from 'react';
import axios from 'axios';
import { API_URL } from '../apiConfig';

export default function Login({ setAuth }) {
    const [user, setUser] = useState('');
    const [pass, setPass] = useState('');
    const [errorMsg, setErrorMsg] = useState('');

    const handleSubmit = async (e) => {
        e.preventDefault();
        setErrorMsg('');
        try {
            const res = await axios.post(`${API_URL}/login`, {
                username: user, 
                password: pass 
            });
            if (res.data.status === 'success') {
                localStorage.setItem('isAuthenticated', 'true');
                setAuth(true);
            }
        } catch (err) {
            setErrorMsg(err.response?.data?.message || "Eroare de conexiune la server.");
        }
    };

    return (
        <div style={{ 
            display: 'flex', 
            justifyContent: 'center', 
            alignItems: 'center', 
            height: '80vh' 
        }}>
            <div style={{ 
                background: 'white', 
                padding: '40px', 
                borderRadius: '12px', 
                boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)', 
                width: '100%', 
                maxWidth: '400px' 
            }}>
                <h2 style={{ marginTop: 0, marginBottom: '24px', textAlign: 'center', color: '#111827' }}>Autentificare</h2>
                
                {errorMsg && (
                    <div style={{ 
                        padding: '12px', 
                        marginBottom: '20px', 
                        backgroundColor: '#fee2e2', 
                        color: '#991b1b', 
                        borderRadius: '6px', 
                        fontSize: '14px',
                        textAlign: 'center'
                    }}>
                        {errorMsg}
                    </div>
                )}

                <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                    <div>
                        <label style={{ display: 'block', fontSize: '14px', fontWeight: '600', color: '#374151', marginBottom: '8px' }}>
                            Nume Utilizator
                        </label>
                        <input 
                            type="text" 
                            required
                            onChange={e => setUser(e.target.value)} 
                            style={{ 
                                width: '100%', 
                                padding: '12px', 
                                border: '1px solid #d1d5db', 
                                borderRadius: '6px', 
                                boxSizing: 'border-box',
                                outline: 'none'
                            }} 
                        />
                    </div>
                    
                    <div>
                        <label style={{ display: 'block', fontSize: '14px', fontWeight: '600', color: '#374151', marginBottom: '8px' }}>
                            Parola
                        </label>
                        <input 
                            type="password" 
                            required
                            onChange={e => setPass(e.target.value)} 
                            style={{ 
                                width: '100%', 
                                padding: '12px', 
                                border: '1px solid #d1d5db', 
                                borderRadius: '6px', 
                                boxSizing: 'border-box',
                                outline: 'none'
                            }} 
                        />
                    </div>
                    
                    <button 
                        type="submit" 
                        style={{ 
                            padding: '14px', 
                            backgroundColor: '#2563eb', 
                            color: 'white', 
                            border: 'none', 
                            borderRadius: '6px', 
                            fontWeight: 'bold', 
                            cursor: 'pointer',
                            marginTop: '10px'
                        }}
                    >
                        Acces Sistem
                    </button>
                </form>
            </div>
        </div>
    );
}