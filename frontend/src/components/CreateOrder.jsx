import { useState, useEffect } from 'react';
import axios from 'axios';
import { jsPDF } from 'jspdf';
import { API_URL } from '../apiConfig';

const CreateOrder = () => {
    const [products, setProducts] = useState([]);
    const [selected, setSelected] = useState('');
    const [quantity, setQuantity] = useState(1);
    const [customerName, setCustomerName] = useState('');
    const [customerPhone, setCustomerPhone] = useState('');
    const [customerAddress, setCustomerAddress] = useState('');
    const [msg, setMsg] = useState('');
    const [msgType, setMsgType] = useState(''); // 'success' sau 'error'
    const [errors, setErrors] = useState({});

    // Folosim useEffect pentru a incarca datele corect
    useEffect(() => {
        const loadData = async () => {
            try {
                const res = await axios.get(`${API_URL}/products`);
                setProducts(res.data);
            } catch (err) {
                console.error("Eroare la incarcare:", err);
            }
        };
        
        loadData();
    }, []); // Array gol = ruleaza doar la montare

    const selectedProductObj = products.find(p => p.id === parseInt(selected));

    const generatePDF = () => {
        const doc = new jsPDF();
        
        doc.setFontSize(22);
        doc.text("Detalii Vanzare", 20, 20);
        
        doc.setFontSize(12);
        doc.text(`Data Comenzii: ${new Date().toLocaleString()}`, 20, 35);
        
        doc.setFontSize(14);
        doc.text("Date Client:", 20, 50);
        doc.setFontSize(12);
        doc.text(`Nume: ${customerName}`, 20, 60);
        doc.text(`Telefon: ${customerPhone}`, 20, 70);
        doc.text(`Adresa: ${customerAddress}`, 20, 80);
        
        doc.text("---------------------------------------------------------", 20, 90);
        
        doc.setFontSize(14);
        doc.text("Detalii Produse:", 20, 105);
        doc.setFontSize(12);
        doc.text(`Produs: ${selectedProductObj.name}`, 20, 115);
        doc.text(`Cantitate Achizitionata: ${quantity} buc`, 20, 125);
        doc.text(`Pret Unitar (Baza): ${selectedProductObj.base_cost} RON`, 20, 135);
        
        doc.setFontSize(14);
        doc.text(`TOTAL DE PLATA: ${selectedProductObj.base_cost * quantity} RON`, 20, 155);
        
        doc.save(`Vanzare_${customerName.replace(/\s+/g, '_')}_${Date.now()}.pdf`);
    };

    const sell = async () => {
        const newErrors = {};

        if (!customerName.trim() || customerName.trim().length < 3)
            newErrors.customerName = "Numele trebuie sa aiba minim 3 caractere.";
        else if (!/^[a-zA-ZăâîșțĂÂÎȘȚ\s-]+$/.test(customerName))
            newErrors.customerName = "Numele nu poate contine cifre sau caractere speciale.";

        if (customerPhone && !/^[0-9\s+\-().]+$/.test(customerPhone))
            newErrors.customerPhone = "Telefonul poate contine doar cifre, spatii si +.";
        else if (customerPhone && customerPhone.replace(/\D/g, '').length < 10)
            newErrors.customerPhone = "Numarul de telefon trebuie sa aiba minim 10 cifre.";

        if (customerAddress && customerAddress.trim().length < 5)
            newErrors.customerAddress = "Adresa trebuie sa aiba minim 5 caractere.";

        if (!selected)
            newErrors.selected = "Selecteaza un produs.";

        if (!quantity || parseInt(quantity) < 1)
            newErrors.quantity = "Cantitatea trebuie sa fie cel putin 1.";
        else if (parseInt(quantity) > selectedProductObj?.stock_quantity)
            newErrors.quantity = `Stoc insuficient. Maxim disponibil: ${selectedProductObj?.stock_quantity} buc.`;

        setErrors(newErrors);
        if (Object.keys(newErrors).length > 0) {
            setMsgType('error');
            setMsg("Corecteaza erorile marcate mai jos.");
            return;
        }
        
        try {
            const res = await axios.post(`${API_URL}/api/orders`, {
                furniture_id: selected, 
                quantity: parseInt(quantity),
                customer_name: customerName
            });
            
            setMsgType('success');
            setMsg(`Vandut! Stoc ramas: ${res.data.new_stock}`);
            generatePDF();
            setSelected('');
            setQuantity(1);
            setCustomerName('');
            setCustomerPhone('');
            setCustomerAddress('');
            setErrors({});

            // Reincarcam lista pentru a vedea noul stoc imediat
            const refresh = await axios.get(`${API_URL}/products`);
            setProducts(refresh.data);
        } catch (err) {
            setMsgType('error');
            setMsg("Eroare: " + (err.response?.data?.message || "Server offline"));
        }
    };

    return (
        <div style={{ padding: '20px', maxWidth: '600px', margin: 'auto' }}>
            <div style={{ background: 'white', padding: '30px', borderRadius: '10px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)' }}>
                <h2 style={{ marginTop: 0, marginBottom: '20px', color: '#1f2937' }}>Efectuare Vanzare Noua</h2>
                
                {msg && <p style={{ 
                    padding: '12px', 
                    backgroundColor: msgType === 'error' ? '#fee2e2' : '#d1fae5',
                    color: msgType === 'error' ? '#991b1b' : '#065f46',
                    borderRadius: '5px' 
                }}>{msg}</p>}
                
                <div style={{ display: 'flex', flexDirection: 'column', gap: '15px', marginBottom: '25px' }}>

                    <div>
                        <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>Nume Client *</label>
                        <input type="text" value={customerName} onChange={e => { setCustomerName(e.target.value); setErrors(prev => ({ ...prev, customerName: null })); }}
                            style={{ width: '100%', padding: '10px', boxSizing: 'border-box', borderRadius: '5px', border: `1px solid ${errors.customerName ? '#ef4444' : '#ccc'}` }}
                            placeholder="Ex: Ion Popescu" />
                        {errors.customerName && <p style={{ color: '#ef4444', fontSize: '13px', margin: '4px 0 0' }}>{errors.customerName}</p>}
                    </div>

                    <div>
                        <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>Telefon</label>
                        <input type="text" value={customerPhone} onChange={e => { setCustomerPhone(e.target.value); setErrors(prev => ({ ...prev, customerPhone: null })); }}
                            style={{ width: '100%', padding: '10px', boxSizing: 'border-box', borderRadius: '5px', border: `1px solid ${errors.customerPhone ? '#ef4444' : '#ccc'}` }}
                            placeholder="Ex: 0712 345 678" />
                        {errors.customerPhone && <p style={{ color: '#ef4444', fontSize: '13px', margin: '4px 0 0' }}>{errors.customerPhone}</p>}
                    </div>

                    <div>
                        <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>Adresa de Livrare</label>
                        <input type="text" value={customerAddress} onChange={e => { setCustomerAddress(e.target.value); setErrors(prev => ({ ...prev, customerAddress: null })); }}
                            style={{ width: '100%', padding: '10px', boxSizing: 'border-box', borderRadius: '5px', border: `1px solid ${errors.customerAddress ? '#ef4444' : '#ccc'}` }}
                            placeholder="Str. Lalelelor, nr. 1, Bucuresti" />
                        {errors.customerAddress && <p style={{ color: '#ef4444', fontSize: '13px', margin: '4px 0 0' }}>{errors.customerAddress}</p>}
                    </div>

                    <hr style={{ border: 'none', borderTop: '1px solid #e5e7eb', margin: '10px 0' }} />

                    <div>
                        <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>Selecteaza Produsul *</label>
                        <select onChange={e => { setSelected(e.target.value); setErrors(prev => ({ ...prev, selected: null })); }} value={selected}
                            style={{ width: '100%', padding: '10px', boxSizing: 'border-box', borderRadius: '5px', border: `1px solid ${errors.selected ? '#ef4444' : '#ccc'}` }}>
                            <option value="">-- Alege Produsul --</option>
                            {products.map(p => (
                                <option key={p.id} value={p.id}>{p.name} - {p.base_cost} RON (Stoc: {p.stock_quantity})</option>
                            ))}
                        </select>
                        {errors.selected && <p style={{ color: '#ef4444', fontSize: '13px', margin: '4px 0 0' }}>{errors.selected}</p>}
                    </div>

                    <div>
                        <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>Cantitate *</label>
                        <input type="number" min="1" max={selectedProductObj ? selectedProductObj.stock_quantity : 1}
                            value={quantity} onChange={e => { setQuantity(e.target.value); setErrors(prev => ({ ...prev, quantity: null })); }}
                            style={{ width: '100%', padding: '10px', boxSizing: 'border-box', borderRadius: '5px', border: `1px solid ${errors.quantity ? '#ef4444' : '#ccc'}` }} />
                        {errors.quantity && <p style={{ color: '#ef4444', fontSize: '13px', margin: '4px 0 0' }}>{errors.quantity}</p>}
                    </div>

                </div>

                <button 
                    onClick={sell} 
                    style={{ 
                        width: '100%', 
                        padding: '14px', 
                        background: '#10b981', 
                        color: 'white', 
                        border: 'none', 
                        borderRadius: '5px',
                        cursor: 'pointer',
                        fontWeight: 'bold',
                        fontSize: '16px'
                    }}
                >
                    Efectueaza Vanzarea & Descarca PDF
                </button>
            </div>
        </div>
    );
};

export default CreateOrder;