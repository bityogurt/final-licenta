import { useState, useEffect } from 'react';
import axios from 'axios';
import { API_URL } from '../apiConfig';

const CATEGORIES = ['Chairs', 'Sofas & armchairs', 'Tables & desks'];
const MATERIALS  = ['Lemn Masiv', 'MDF/PAL', 'Metal', 'Necunoscut'];
const UPHOLSTERY = ['Fara', 'Piele', 'Textil', 'Necunoscut'];

const CAT_RO = {
    'Chairs':            'Scaune',
    'Sofas & armchairs': 'Canapele & Fotolii',
    'Tables & desks':    'Mese & Birouri',
};
const CAT_COLOR = {
    'Chairs':            { bg: '#dbeafe', color: '#1d4ed8' },
    'Sofas & armchairs': { bg: '#ede9fe', color: '#6d28d9' },
    'Tables & desks':    { bg: '#dcfce7', color: '#166534' },
};

const emptyForm = {
    name: '', category: 'Chairs', width_cm: '', depth_cm: '', height_cm: '',
    material_structure: 'Lemn Masiv', upholstery_material: 'Fara',
    is_extensible: false, seat_count: 0, table_capacity_persons: 0,
    base_cost: '', selling_price: '', stock_quantity: 0,
};

export default function Inventory() {
    const [products, setProducts]           = useState([]);
    const [form, setForm]                   = useState(emptyForm);
    const [msg, setMsg]                     = useState('');
    const [editingStock, setEditingStock]   = useState(null);

    // tab si filtre
    const [activeTab, setActiveTab]           = useState('add');
    const [filterCategory, setFilterCategory] = useState('toate');
    const [searchName, setSearchName]         = useState('');

    // re-antrenare + actualizare preturi
    const [retraining, setRetraining]         = useState(false);
    const [retrainResult, setRetrainResult]   = useState(null);
    const [updatingPrices, setUpdatingPrices] = useState(false);
    const [updateMsg, setUpdateMsg]           = useState('');

    const fetchData = async () => {
        const res = await axios.get(`${API_URL}/api/inventory`);
        setProducts(res.data.products);
    };

    useEffect(() => { fetchData(); }, []); // eslint-disable-line react-hooks/exhaustive-deps

    const set = (field, value) => setForm(f => ({ ...f, [field]: value }));

    const handleCategoryChange = (cat) => {
        setForm(f => ({
            ...f,
            category: cat,
            upholstery_material:
                cat === 'Tables & desks' ? 'Fara'
                : cat === 'Sofas & armchairs' && f.upholstery_material === 'Fara' ? 'Textil'
                : f.upholstery_material,
            seat_count:             cat === 'Chairs' ? 1
                                  : cat === 'Tables & desks' ? 0
                                  : f.seat_count,
            table_capacity_persons: cat !== 'Tables & desks' ? 0 : f.table_capacity_persons,
            is_extensible:          cat !== 'Tables & desks' ? false : f.is_extensible,
        }));
    };

    const upholsteryOptions = form.category === 'Sofas & armchairs'
        ? ['Piele', 'Textil', 'Necunoscut']
        : UPHOLSTERY;

    const showUpholstery = form.category !== 'Tables & desks';
    const showSeats      = form.category !== 'Tables & desks';
    const showTableCap   = form.category === 'Tables & desks';

    const isDuplicateName = form.name.trim() &&
        products.some(p => p.name.toLowerCase().trim() === form.name.toLowerCase().trim());

    const handleDelete = async (id, name) => {
        if (!window.confirm(`Stergi produsul "${name}" din inventar?`)) return;
        await axios.delete(`${API_URL}/api/furniture/${id}`);
        fetchData();
    };

    const saveStock = async (id) => {
        const val = parseInt(editingStock.value);
        if (isNaN(val) || val < 0) { setEditingStock(null); return; }
        await axios.patch(`${API_URL}/api/furniture/${id}/stock`, { stock_quantity: val });
        setEditingStock(null);
        fetchData();
    };

    const handleAddProduct = async (e) => {
        e.preventDefault();
        setMsg('');
        if (isDuplicateName) {
            setMsg('Eroare: Exista deja un produs cu acest nume in inventar.');
            return;
        }
        try {
            await axios.post(`${API_URL}/api/furniture`, {
                ...form,
                is_extensible: form.is_extensible ? 1 : 0,
            });
            setMsg('Produs adaugat cu succes. Pretul ML a fost calculat automat.');
            setForm(emptyForm);
            fetchData();
        } catch (err) {
            setMsg('Eroare la salvare: ' + (err.response?.data?.message || 'Eroare retea'));
        }
    };

    const handleUpdateAllPrices = async () => {
        setUpdatingPrices(true);
        setUpdateMsg('');
        try {
            const res = await axios.post(`${API_URL}/api/update-all-prices`);
            setUpdateMsg(`${res.data.updated} preturi ML actualizate.`);
            fetchData();
        } catch (err) {
            setUpdateMsg('Eroare: ' + (err.response?.data?.message || 'Eroare retea'));
        } finally {
            setUpdatingPrices(false);
        }
    };

    const handleRetrain = async () => {
        setRetraining(true);
        setRetrainResult(null);
        try {
            const res = await axios.post(`${API_URL}/api/retrain`);
            setRetrainResult(res.data);
            fetchData(); // preturile ML se recalculeaza dupa reantrenare
        } catch (err) {
            setRetrainResult({ error: err.response?.data?.message || 'Eroare la re-antrenare' });
        } finally {
            setRetraining(false);
        }
    };

    const filteredProducts = products
        .filter(p => filterCategory === 'toate' || p.category === filterCategory)
        .filter(p => p.name.toLowerCase().includes(searchName.toLowerCase()));

    const lowStockCount    = products.filter(p => p.stock_quantity < 5).length;
    const withSellingPrice = products.filter(p => p.selling_price).length;
    const missingMlPrice   = products.filter(p => !p.suggested_price).length;

    const inputStyle ={ width: '100%', padding: '8px', boxSizing: 'border-box', borderRadius: '4px', border: '1px solid #d1d5db' };
    const labelStyle = { display: 'block', fontWeight: 'bold', marginBottom: '5px', fontSize: '0.875rem' };
    const fieldStyle = { marginBottom: '12px' };

    const tabBtn = (id, label) => (
        <button onClick={() => setActiveTab(id)} style={{
            padding: '11px 24px', border: 'none',
            borderBottom: activeTab === id ? '2px solid #2563eb' : '2px solid transparent',
            background: 'transparent',
            color: activeTab === id ? '#2563eb' : '#6b7280',
            fontWeight: activeTab === id ? '700' : '400',
            cursor: 'pointer', fontSize: '0.95rem',
        }}>
            {label}
        </button>
    );

    return (
        <div style={{ padding: '20px', maxWidth: '1400px', margin: '0 auto' }}>

            {/* Tab bar */}
            <div style={{ display: 'flex', borderBottom: '2px solid #e5e7eb', marginBottom: '24px', gap: '4px' }}>
                {tabBtn('add', '+ Adauga Produs Nou')}
                {tabBtn('view',
                    <span>
                        Produse Existente ({products.length})
                        {lowStockCount > 0 && (
                            <span style={{ marginLeft: '8px', padding: '2px 7px', background: '#fee2e2', color: '#dc2626', borderRadius: '10px', fontSize: '0.72rem', fontWeight: 'bold' }}>
                                {lowStockCount} stoc critic
                            </span>
                        )}
                    </span>
                )}
            </div>

            {/* tab: adauga produs */}
            {activeTab === 'add' && (
                <div style={{ display: 'flex', justifyContent: 'center' }}>
                    <div style={{ width: '100%', maxWidth: '520px', background: 'white', padding: '24px', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
                        <h3 style={{ marginTop: 0 }}>Adauga Produs Nou</h3>
                        <hr style={{ marginBottom: '16px' }} />

                        {msg && (
                            <div style={{
                                padding: '10px', marginBottom: '12px', borderRadius: '4px',
                                backgroundColor: msg.includes('Eroare') ? '#fee2e2' : '#d1fae5',
                                color: msg.includes('Eroare') ? '#991b1b' : '#065f46',
                            }}>
                                {msg}
                            </div>
                        )}

                        <form onSubmit={handleAddProduct} style={{ display: 'flex', flexDirection: 'column' }}>

                            {/* Nume */}
                            <div style={fieldStyle}>
                                <label style={labelStyle}>Nume Produs</label>
                                <input
                                    style={{ ...inputStyle, borderColor: isDuplicateName ? '#dc2626' : '#d1d5db' }}
                                    value={form.name}
                                    onChange={e => set('name', e.target.value)}
                                    required
                                />
                                {isDuplicateName && (
                                    <span style={{ fontSize: '0.75rem', color: '#dc2626', marginTop: '3px', display: 'block' }}>
                                        ⚠ Exista deja un produs cu acest nume
                                    </span>
                                )}
                            </div>

                            {/* Categorie */}
                            <div style={fieldStyle}>
                                <label style={labelStyle}>Categorie</label>
                                <select style={inputStyle} value={form.category} onChange={e => handleCategoryChange(e.target.value)}>
                                    {CATEGORIES.map(c => <option key={c}>{c}</option>)}
                                </select>
                            </div>

                            {/* Dimensiuni */}
                            <div style={{ display: 'flex', gap: '8px', marginBottom: '12px' }}>
                                {[['Latime (cm)', 'width_cm'], ['Adancime (cm)', 'depth_cm'], ['Inaltime (cm)', 'height_cm']].map(([lbl, field]) => (
                                    <div key={field} style={{ flex: 1 }}>
                                        <label style={labelStyle}>{lbl}</label>
                                        <input style={inputStyle} type="number" step="0.1"
                                            value={form[field]} onChange={e => set(field, e.target.value)} required />
                                    </div>
                                ))}
                            </div>

                            {/* Material + Tapiterie */}
                            <div style={{ display: 'flex', gap: '8px', marginBottom: '12px' }}>
                                <div style={{ flex: showUpholstery ? 1 : undefined, width: showUpholstery ? undefined : '100%' }}>
                                    <label style={labelStyle}>Material Structura</label>
                                    <select style={inputStyle} value={form.material_structure} onChange={e => set('material_structure', e.target.value)}>
                                        {MATERIALS.map(m => <option key={m}>{m}</option>)}
                                    </select>
                                </div>
                                {showUpholstery && (
                                    <div style={{ flex: 1 }}>
                                        <label style={labelStyle}>Tapiterie</label>
                                        <select style={inputStyle} value={form.upholstery_material} onChange={e => set('upholstery_material', e.target.value)}>
                                            {upholsteryOptions.map(u => <option key={u}>{u}</option>)}
                                        </select>
                                    </div>
                                )}
                            </div>

                            {/* Locuri / Capacitate masa / Extensibil — conditionat de categorie */}
                            {(showSeats || showTableCap) && (
                                <div style={{ display: 'flex', gap: '8px', alignItems: 'flex-end', marginBottom: '12px' }}>
                                    {showSeats && (
                                        <div style={{ flex: 1 }}>
                                            <label style={labelStyle}>Nr. Locuri</label>
                                            <input style={inputStyle} type="number" min="0" value={form.seat_count} onChange={e => set('seat_count', e.target.value)} />
                                        </div>
                                    )}
                                    {showTableCap && (
                                        <div style={{ flex: 1 }}>
                                            <label style={labelStyle}>Nr. Locuri Masă</label>
                                            <input style={inputStyle} type="number" min="0" value={form.table_capacity_persons} onChange={e => set('table_capacity_persons', e.target.value)} />
                                        </div>
                                    )}
                                    {showTableCap && (
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', paddingBottom: '8px' }}>
                                            <input type="checkbox" id="extensibil" checked={form.is_extensible} onChange={e => set('is_extensible', e.target.checked)} />
                                            <label htmlFor="extensibil" style={{ fontWeight: 'bold', fontSize: '0.875rem', whiteSpace: 'nowrap' }}>Extensibila</label>
                                        </div>
                                    )}
                                </div>
                            )}

                            {/* Costuri + Stoc */}
                            <div style={{ display: 'flex', gap: '8px', marginBottom: '16px' }}>
                                <div style={{ flex: 1 }}>
                                    <label style={labelStyle}>Cost Productie (RON)</label>
                                    <input style={inputStyle} type="number" step="0.01" value={form.base_cost} onChange={e => set('base_cost', e.target.value)} />
                                </div>
                                <div style={{ flex: 1 }}>
                                    <label style={labelStyle}>Pret Vanzare (RON)</label>
                                    <input style={inputStyle} type="number" step="0.01" value={form.selling_price} onChange={e => set('selling_price', e.target.value)} />
                                </div>
                                <div style={{ flex: 1 }}>
                                    <label style={labelStyle}>Stoc Initial</label>
                                    <input style={inputStyle} type="number" min="0" value={form.stock_quantity} onChange={e => set('stock_quantity', e.target.value)} />
                                </div>
                            </div>

                            <div style={{ padding: '10px 12px', background: '#f0f9ff', borderRadius: '6px', marginBottom: '14px', fontSize: '0.82rem', color: '#0e7490' }}>
                                Pretul ML va fi calculat automat la salvare pe baza dimensiunilor si categoriei produsului.
                            </div>

                            <button type="submit" disabled={isDuplicateName}
                                style={{
                                    padding: '11px', background: isDuplicateName ? '#93c5fd' : '#2563eb',
                                    color: 'white', border: 'none', borderRadius: '4px', fontWeight: 'bold',
                                    cursor: isDuplicateName ? 'not-allowed' : 'pointer', fontSize: '0.95rem',
                                }}>
                                Salveaza Produsul in Inventar
                            </button>
                        </form>
                    </div>
                </div>
            )}

            {/* tab: produse existente */}
            {activeTab === 'view' && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>

                    {/* Panel ML */}
                    <div style={{ background: 'white', padding: '16px 20px', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
                            <span style={{ fontWeight: 'bold', fontSize: '0.9rem' }}>Model ML — Random Forest</span>
                            <span style={{ color: '#6b7280', fontSize: '0.82rem' }}>
                                {missingMlPrice > 0 ? `${missingMlPrice} produs${missingMlPrice > 1 ? 'e' : ''} fara pret ML` : 'Toate produsele au pret ML calculat'}
                                {withSellingPrice > 0 && ` · ${withSellingPrice} cu pret de vanzare (disponibile pentru re-antrenare)`}
                            </span>
                            <div style={{ marginLeft: 'auto', display: 'flex', gap: '8px' }}>
                                <button onClick={handleUpdateAllPrices} disabled={updatingPrices}
                                    style={{
                                        padding: '7px 14px', background: updatingPrices ? '#e5e7eb' : '#0891b2',
                                        color: updatingPrices ? '#9ca3af' : 'white', border: 'none', borderRadius: '6px',
                                        fontWeight: '600', cursor: updatingPrices ? 'not-allowed' : 'pointer', fontSize: '0.82rem',
                                    }}>
                                    {updatingPrices ? 'Se calculeaza...' : 'Actualizeaza Preturile ML'}
                                </button>
                                <button onClick={handleRetrain} disabled={retraining}
                                    style={{
                                        padding: '7px 14px', background: retraining ? '#e5e7eb' : '#7c3aed',
                                        color: retraining ? '#9ca3af' : 'white', border: 'none', borderRadius: '6px',
                                        fontWeight: '600', cursor: retraining ? 'not-allowed' : 'pointer', fontSize: '0.82rem',
                                    }}>
                                    {retraining ? 'Se antreneaza...' : 'Reantreneaza Modelul'}
                                </button>
                            </div>
                        </div>

                        {updateMsg && (
                            <div style={{ marginTop: '10px', padding: '8px 12px', borderRadius: '4px', background: '#d1fae5', color: '#065f46', fontSize: '0.82rem' }}>
                                {updateMsg}
                            </div>
                        )}

                        {retrainResult && (
                            <div style={{
                                marginTop: '10px', padding: '10px 14px', borderRadius: '6px',
                                background: retrainResult.error ? '#fee2e2' : '#d1fae5',
                                color: retrainResult.error ? '#991b1b' : '#065f46', fontSize: '0.875rem',
                            }}>
                                {retrainResult.error ? retrainResult.error : (
                                    <>
                                        <strong>Reantrenare finalizata!</strong>
                                        {' '}R² = <strong>{retrainResult.r2_score}</strong>
                                        {' '}| MAE = <strong>{retrainResult.mae_ron} RON</strong>
                                        <span style={{ marginLeft: '10px', opacity: 0.8 }}>
                                            ({retrainResult.ikea_samples} IKEA + {retrainResult.user_samples} proprii)
                                        </span>
                                    </>
                                )}
                            </div>
                        )}
                    </div>

                    {/* Bara filtrare */}
                    <div style={{ background: 'white', padding: '14px 20px', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)', display: 'flex', gap: '12px', alignItems: 'center', flexWrap: 'wrap' }}>
                        <input
                            placeholder="Cauta dupa nume..."
                            value={searchName}
                            onChange={e => setSearchName(e.target.value)}
                            style={{ padding: '7px 12px', border: '1px solid #d1d5db', borderRadius: '4px', minWidth: '200px', fontSize: '0.875rem' }}
                        />
                        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                            {['toate', ...CATEGORIES].map(cat => {
                                const active = filterCategory === cat;
                                const c = cat !== 'toate' ? CAT_COLOR[cat] : null;
                                return (
                                    <button key={cat} onClick={() => setFilterCategory(cat)} style={{
                                        padding: '5px 14px', border: '1px solid',
                                        borderColor: active ? (c?.color || '#2563eb') : '#d1d5db',
                                        background: active ? (c?.bg || '#dbeafe') : 'white',
                                        color: active ? (c?.color || '#1d4ed8') : '#6b7280',
                                        borderRadius: '20px', cursor: 'pointer', fontSize: '0.8rem',
                                        fontWeight: active ? '700' : '400',
                                    }}>
                                        {cat === 'toate' ? 'Toate' : CAT_RO[cat] || cat}
                                        <span style={{ marginLeft: '4px', opacity: 0.75 }}>
                                            ({cat === 'toate' ? products.length : products.filter(p => p.category === cat).length})
                                        </span>
                                    </button>
                                );
                            })}
                        </div>
                        <span style={{ marginLeft: 'auto', color: '#6b7280', fontSize: '0.8rem' }}>
                            {filteredProducts.length} produse
                        </span>
                    </div>

                    {/* Tabel produse finite */}
                    <div style={{ background: 'white', padding: '20px', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
                        <h3 style={{ marginTop: 0, marginBottom: '16px' }}>Stoc Produse Finite</h3>
                        <div style={{ overflowX: 'auto' }}>
                            <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.85rem' }}>
                                <thead>
                                    <tr style={{ backgroundColor: '#f9fafb', borderBottom: '2px solid #e5e7eb' }}>
                                        <th style={{ padding: '10px 12px' }}>Nume</th>
                                        <th style={{ padding: '10px 12px' }}>Categorie</th>
                                        <th style={{ padding: '10px 12px' }}>Dim. L×A×H (cm)</th>
                                        <th style={{ padding: '10px 12px' }}>Material</th>
                                        <th style={{ padding: '10px 12px' }}>Tapiterie</th>
                                        <th style={{ padding: '10px 12px' }}>Cost Prod.</th>
                                        <th style={{ padding: '10px 12px' }}>Pret Vanzare</th>
                                        <th style={{ padding: '10px 12px', color: '#7c3aed' }}>Pret ML</th>
                                        <th style={{ padding: '10px 12px' }}>Stoc</th>
                                        <th style={{ padding: '10px 12px' }}></th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {filteredProducts.length === 0 ? (
                                        <tr>
                                            <td colSpan={9} style={{ padding: '24px', textAlign: 'center', color: '#9ca3af' }}>
                                                Niciun produs gasit.
                                            </td>
                                        </tr>
                                    ) : filteredProducts.map((p, i) => {
                                        const catStyle = CAT_COLOR[p.category] || { bg: '#f3f4f6', color: '#374151' };
                                        const isLow    = p.stock_quantity < 5;
                                        return (
                                            <tr key={p.id} style={{ borderBottom: '1px solid #e5e7eb', backgroundColor: i % 2 === 0 ? 'white' : '#fafafa' }}>
                                                <td style={{ padding: '10px 12px', fontWeight: '600' }}>{p.name}</td>
                                                <td style={{ padding: '10px 12px' }}>
                                                    <span style={{ padding: '3px 10px', borderRadius: '12px', background: catStyle.bg, color: catStyle.color, fontWeight: '600', fontSize: '0.75rem', whiteSpace: 'nowrap' }}>
                                                        {CAT_RO[p.category] || p.category}
                                                    </span>
                                                </td>
                                                <td style={{ padding: '10px 12px', color: '#6b7280', fontFamily: 'monospace' }}>
                                                    {p.width_cm}×{p.depth_cm}×{p.height_cm}
                                                </td>
                                                <td style={{ padding: '10px 12px' }}>{p.material_structure}</td>
                                                <td style={{ padding: '10px 12px', color: p.upholstery_material === 'Fara' ? '#9ca3af' : '#374151' }}>
                                                    {p.upholstery_material}
                                                </td>
                                                <td style={{ padding: '10px 12px' }}>
                                                    {p.base_cost ? `${p.base_cost} RON` : <span style={{ color: '#9ca3af' }}>—</span>}
                                                </td>
                                                <td style={{ padding: '10px 12px', fontWeight: '600', color: p.selling_price ? '#065f46' : '#9ca3af' }}>
                                                    {p.selling_price ? `${p.selling_price} RON` : '—'}
                                                </td>
                                                <td style={{ padding: '10px 12px' }}>
                                                    {p.suggested_price ? (
                                                        <span style={{ fontWeight: '700', color: '#7c3aed' }}>{p.suggested_price} RON</span>
                                                    ) : (
                                                        <span style={{ color: '#9ca3af', fontSize: '0.8rem' }}>necalculat</span>
                                                    )}
                                                </td>
                                                <td style={{ padding: '10px 12px' }}>
                                                    {editingStock?.id === p.id ? (
                                                        <input type="number" min="0" value={editingStock.value} autoFocus
                                                            onChange={e => setEditingStock({ id: p.id, value: e.target.value })}
                                                            onBlur={() => saveStock(p.id)}
                                                            onKeyDown={e => { if (e.key === 'Enter') saveStock(p.id); if (e.key === 'Escape') setEditingStock(null); }}
                                                            style={{ width: '70px', padding: '4px', fontWeight: 'bold' }}
                                                        />
                                                    ) : (
                                                        <span onClick={() => setEditingStock({ id: p.id, value: p.stock_quantity })}
                                                            title="Click pentru editare stoc"
                                                            style={{
                                                                display: 'inline-block', padding: '3px 10px', borderRadius: '12px',
                                                                fontWeight: '700', fontSize: '0.8rem', cursor: 'pointer',
                                                                background: isLow ? '#fee2e2' : '#dcfce7',
                                                                color: isLow ? '#dc2626' : '#16a34a',
                                                                border: `1px dashed ${isLow ? '#dc2626' : '#16a34a'}`,
                                                            }}>
                                                            {p.stock_quantity} buc{isLow ? ' ⚠' : ''}
                                                        </span>
                                                    )}
                                                </td>
                                                <td style={{ padding: '10px 12px' }}>
                                                    <button
                                                        onClick={() => handleDelete(p.id, p.name)}
                                                        title="Sterge produs"
                                                        style={{
                                                            background: 'none', border: '1px solid #fca5a5',
                                                            color: '#dc2626', borderRadius: '4px',
                                                            padding: '3px 8px', cursor: 'pointer', fontSize: '0.8rem',
                                                        }}
                                                    >
                                                        Sterge
                                                    </button>
                                                </td>
                                            </tr>
                                        );
                                    })}
                                </tbody>
                            </table>
                        </div>
                    </div>

                </div>
            )}
        </div>
    );
}
