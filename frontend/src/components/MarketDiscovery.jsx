import { useState, useEffect } from 'react';
import axios from 'axios';
import { API_URL } from '../apiConfig';

const MarketDiscovery = () => {
    const [products, setProducts]           = useState([]);
    const [selectedProduct, setSelectedProduct] = useState('');
    const [competitorUrl, setCompetitorUrl] = useState('');
    const [analyzing, setAnalyzing]         = useState(false);
    const [analyzeMsg, setAnalyzeMsg]       = useState('');
    const [result, setResult]               = useState(null);
    const [saving, setSaving]               = useState(false);
    const [saveMsg, setSaveMsg]             = useState('');

    useEffect(() => {
        axios.get(`${API_URL}/products`)
            .then(res => setProducts(res.data))
            .catch(() => {});
    }, []); // eslint-disable-line react-hooks/exhaustive-deps

    const selectedProductObj = products.find(p => p.id === parseInt(selectedProduct));

    const handleGoogleSearch = () => {
        if (!selectedProductObj) { alert('Selecteaza un produs din lista.'); return; }
        window.open(
            `https://www.google.com/search?q=${encodeURIComponent(selectedProductObj.name + ' pret romania')}`,
            '_blank'
        );
    };

    const handleAnalyze = async () => {
        const url = competitorUrl.trim();
        if (!url) return;
        setAnalyzing(true);
        setResult(null);
        setAnalyzeMsg('');
        setSaveMsg('');
        try {
            const res = await axios.post(`${API_URL}/api/scrape-product`, { url });
            const d = res.data.data;
            setResult({ name: d.name || '', price: d.price ?? '' });
            setAnalyzeMsg(res.data.message);
        } catch (err) {
            setAnalyzeMsg('Eroare la analiza: ' + (err.response?.data?.message || 'Eroare retea'));
        } finally {
            setAnalyzing(false);
        }
    };

    const handleSave = async () => {
        if (!result) return;
        setSaving(true);
        setSaveMsg('');
        const saveName = `[AI] ${result.name || (selectedProductObj ? selectedProductObj.name : 'Produs')}`;
        try {
            const res = await axios.post(`${API_URL}/api/import-scraped-product`, {
                name:                saveName,
                category:            selectedProductObj?.category || 'Chairs',
                selling_price:       result.price || null,
                material_structure:  'Necunoscut',
                upholstery_material: 'Fara',
            });
            setSaveMsg(res.data.message);
            setResult(null);
            setCompetitorUrl('');
        } catch (err) {
            setSaveMsg('Eroare: ' + (err.response?.data?.message || 'Eroare retea'));
        } finally {
            setSaving(false);
        }
    };

    const inputStyle = {
        width: '100%', padding: '9px 12px', boxSizing: 'border-box',
        borderRadius: '6px', border: '1px solid #d1d5db', fontSize: '0.875rem',
    };
    const labelStyle = {
        display: 'block', fontWeight: '600', marginBottom: '5px',
        fontSize: '0.8rem', color: '#374151',
    };

    return (
        <div style={{ padding: '24px', maxWidth: '660px', margin: '0 auto', fontFamily: 'system-ui, sans-serif' }}>
            <h2 style={{ marginTop: 0, marginBottom: '6px' }}>Market Discovery</h2>
            <p style={{ margin: '0 0 24px', color: '#6b7280', fontSize: '0.875rem' }}>
                Cauta produse la competitori, extrage pretul cu AI si salveaza-le ca date de antrenare ML.
            </p>

            {/* ---- Card principal: analiza cu AI ---- */}
            <div style={{
                background: 'white', padding: '24px', borderRadius: '8px',
                boxShadow: '0 2px 4px rgba(0,0,0,0.1)', marginBottom: '20px',
            }}>
                <h3 style={{ marginTop: 0, marginBottom: '16px', fontSize: '1rem' }}>
                    Analizeaza Produs Competitor
                </h3>

                {/* Selectare produs de referinta */}
                <div style={{ marginBottom: '12px' }}>
                    <label style={labelStyle}>Produs de referinta din inventar</label>
                    <select
                        style={inputStyle}
                        value={selectedProduct}
                        onChange={e => { setSelectedProduct(e.target.value); setResult(null); setSaveMsg(''); }}
                    >
                        <option value="">-- Alege un produs --</option>
                        {products.map(p => (
                            <option key={p.id} value={p.id}>{p.name}</option>
                        ))}
                    </select>
                </div>

                {/* Buton Google */}
                <button
                    onClick={handleGoogleSearch}
                    disabled={!selectedProduct}
                    style={{
                        width: '100%', padding: '9px', marginBottom: '16px',
                        background: selectedProduct ? '#eff6ff' : '#f3f4f6',
                        color: selectedProduct ? '#2563eb' : '#9ca3af',
                        border: `1px solid ${selectedProduct ? '#3b82f6' : '#d1d5db'}`,
                        borderRadius: '6px', fontWeight: '600',
                        cursor: selectedProduct ? 'pointer' : 'not-allowed', fontSize: '0.875rem',
                    }}
                >
                    Cauta pe Google ↗
                </button>

                {/* URL competitor + analiza */}
                <div style={{ marginBottom: '14px' }}>
                    <label style={labelStyle}>URL pagina competitor</label>
                    <div style={{ display: 'flex', gap: '8px' }}>
                        <input
                            style={{ ...inputStyle, flex: 1 }}
                            type="url"
                            placeholder="https://www.emag.ro/scaun-..."
                            value={competitorUrl}
                            onChange={e => { setCompetitorUrl(e.target.value); setResult(null); setSaveMsg(''); }}
                            onKeyDown={e => e.key === 'Enter' && !analyzing && handleAnalyze()}
                        />
                        <button
                            onClick={handleAnalyze}
                            disabled={analyzing || !competitorUrl.trim()}
                            style={{
                                padding: '9px 18px', whiteSpace: 'nowrap',
                                background: (analyzing || !competitorUrl.trim()) ? '#9ca3af' : '#2563eb',
                                color: 'white', border: 'none', borderRadius: '6px',
                                fontWeight: '600', fontSize: '0.875rem',
                                cursor: (analyzing || !competitorUrl.trim()) ? 'not-allowed' : 'pointer',
                            }}
                        >
                            {analyzing ? 'Se analizeaza...' : 'Analizeaza cu AI'}
                        </button>
                    </div>
                </div>

                {analyzeMsg && (
                    <div style={{
                        padding: '8px 12px', marginBottom: '14px', borderRadius: '4px', fontSize: '0.82rem',
                        background: analyzeMsg.includes('succes') ? '#d1fae5' : '#fef3c7',
                        color:      analyzeMsg.includes('succes') ? '#065f46' : '#92400e',
                    }}>
                        {analyzeMsg}
                    </div>
                )}

                {/* Rezultat si salvare */}
                {result && (
                    <>
                        <hr style={{ margin: '16px 0', borderColor: '#e5e7eb' }} />

                        <div style={{ marginBottom: '12px' }}>
                            <label style={labelStyle}>Nume produs gasit</label>
                            <input
                                style={inputStyle}
                                value={result.name}
                                onChange={e => setResult(prev => ({ ...prev, name: e.target.value }))}
                                placeholder="Editeaza daca e necesar"
                            />
                        </div>

                        <div style={{ marginBottom: '18px' }}>
                            <label style={labelStyle}>Pret gasit (RON)</label>
                            <input
                                style={inputStyle}
                                type="number"
                                step="0.01"
                                value={result.price ?? ''}
                                onChange={e => setResult(prev => ({ ...prev, price: e.target.value }))}
                                placeholder="Completeaza manual daca AI nu a gasit"
                            />
                        </div>

                        {saveMsg && (
                            <div style={{
                                padding: '8px 12px', marginBottom: '12px', borderRadius: '4px', fontSize: '0.82rem',
                                background: saveMsg.includes('importat') ? '#d1fae5' : '#fee2e2',
                                color:      saveMsg.includes('importat') ? '#065f46' : '#991b1b',
                            }}>
                                {saveMsg}
                            </div>
                        )}

                        <button
                            onClick={handleSave}
                            disabled={saving}
                            style={{
                                width: '100%', padding: '11px',
                                background: saving ? '#9ca3af' : '#10b981',
                                color: 'white', border: 'none', borderRadius: '6px',
                                fontWeight: '700', fontSize: '0.9rem',
                                cursor: saving ? 'not-allowed' : 'pointer',
                            }}
                        >
                            {saving ? 'Se salveaza...' : 'Salveaza in Baza de Date'}
                        </button>

                        <p style={{ margin: '8px 0 0', fontSize: '0.75rem', color: '#9ca3af' }}>
                            Produsul va fi salvat cu prefixul [AI] si pretul gasit ca pret de vanzare,
                            contribuind la re-antrenarea modelului ML.
                        </p>
                    </>
                )}
            </div>

        </div>
    );
};

export default MarketDiscovery;
