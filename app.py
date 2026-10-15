import os
import random
from flask import Flask, render_template, request, redirect, url_for, jsonify
from ultralytics import YOLO
from google import genai

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 1. Inisialisasi Model YOLOv8
model = YOLO('yolov8n-cls.pt')

# 2. Inisialisasi Google GenAI secara aman menggunakan Environment Variable
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

eco_facts = [
    "💡 Tahukah kamu? Mendaur ulang 1 ton kertas dapat menyelamatkan sekitar 17 pohon besar!",
    "💡 Fakta: Plastik tidak benar-benar terurai, melainkan pecah menjadi partikel mikroplastik di alam.",
    "💡 Tahukah kamu? Kaca adalah material yang dapat didaur ulang selamanya tanpa mengurangi kualitasnya.",
]

# Kamus cadangan lokal jika server AI sedang sibuk (503)
fallback_advice = {
    "pop_bottle": "Botol kaca atau botol minuman bersoda ini dapat didaur ulang. Pastikan dicuci bersih, lepaskan tutup atau labelnya, lalu serahkan ke bank sampah atau tempat pengumpulan.",
    "water bottle": "Botol plastik memiliki nilai daur ulang tinggi. Lepaskan label dan tutupnya, bilas bersih, lalu pipihkan untuk menghemat ruang.",
    "cardboard box": "Kardus dapat didaur ulang dengan mudah! Lipat kardus menjadi rata dan simpan di tempat yang kering.",
}

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    image_path = None
    advice = None
    random_fact = random.choice(eco_facts)

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file:
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Prediksi YOLO
            results = model(file_path)
            r = results[0]
            top1_index = r.probs.top1
            prediction = r.names[top1_index]
            confidence = float(r.probs.top1conf.item()) * 100

            prompt = (
                f"Saya baru saja mengunggah foto sampah yang terdeteksi sebagai '{prediction}' "
                f"dengan tingkat keyakinan {confidence:.2f}%. Berikan penjelasan edukatif yang mendalam, "
                f"tips daur ulang kreatif, serta dampak lingkungannya secara ramah dan profesional seolah-olah "
                f"kamu adalah pakar lingkungan."
            )

            try:
                response = client.models.generate_content(
                    model='gemini-3.8-flash',
                    contents=prompt,
                )
                advice = response.text
            except Exception as e:
                advice = fallback_advice.get(
                    prediction.lower(), 
                    f"Sampah terdeteksi sebagai '{prediction}'. Pastikan untuk membuangnya sesuai jenis materialnya (organik/anorganik) agar lingkungan tetap bersih dan mudah didaur ulang!"
                )

            image_path = url_for('static', filename='uploads/' + filename)

    return render_template('index.html', prediction=prediction, confidence=confidence, image_path=image_path, advice=advice, random_fact=random_fact)

# Route Chat API dengan fallback otomatis
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_msg = data.get('message', '')

    try:
        response = client.models.generate_content(
            model='gemini-3.8-flash',
            contents=f"Kamu adalah Asisten Lingkungan SmartWaste AI yang ramah, interaktif, dan cerdas. Jawablah pertanyaan berikut seputar sampah, daur ulang, atau lingkungan secara natural dan informatif: {user_msg}"
        )
        bot_reply = response.text
    except Exception as e:
        bot_reply = "🌱 Halo! Saat ini server AI sedang mengalami lonjakan kesibukan. Namun sebagai asisten lingkungan, saya sarankan agar kita selalu memilah sampah dari rumah (organik, anorganik, dan B3) untuk mendukung gerakan daur ulang. Silakan coba tanyakan kembali beberapa saat lagi ya!"

    return jsonify({'reply': bot_reply})

if __name__ == '__main__':
    app.run(debug=True, port=5000)