import pandas as pd

# Nama file Excel Anda
file_input = 'Geografi Merchant Template.xlsx'
# Nama kolom yang berisi skor lokasi Anda
kolom_lokasi = 'Penilaian Lokasi (1-5)'

try:
	# Baca file Excel
	df = pd.read_excel(file_input, skiprows=3)
	
	print(df.columns)

	# Pastikan total bobot adalah 1.0 (atau 100%)
	total_bobot = df['Bobot (%)'].sum()
	if not (0.99 <= total_bobot <= 1.01):
		print(f"Peringatan: Total bobot Anda adalah {total_bobot*100:.0f}%, bukan 100%. Harap periksa kembali.")
	else:
		# Hitung nilai dengan mengalikan bobot dan skor
		df['Nilai'] = df['Bobot (%)'] * df[kolom_lokasi]

		# Hitung skor akhir dengan menjumlahkan semua nilai
		skor_akhir = df['Nilai'].sum()

		# Tampilkan hasil
		print("===== HASIL ANALISIS LOKASI USAHA =====")
		print(f"Lokasi: {kolom_lokasi}")
		print(f"Skor Akhir: {skor_akhir:.2f} dari 5.00")
		print("=" * 40)

		# Berikan interpretasi sederhana
		if skor_akhir >= 4.0:
			print("📈 Interpretasi: Skor sangat baik. Lokasi ini memiliki potensi yang sangat kuat.")
		elif skor_akhir >= 3.0:
			print("👍 Interpretasi: Skor baik. Lokasi ini memiliki potensi yang solid.")
		elif skor_akhir >= 2.0:
			print("🤔 Interpretasi: Skor cukup. Ada beberapa aspek yang perlu diwaspadai atau diperbaiki.")
		else:
			print("⚠️ Interpretasi: Skor rendah. Lokasi ini memiliki banyak tantangan, pertimbangkan dengan sangat hati-hati.")

except FileNotFoundError:
	print(f"Error: File '{file_input}' tidak ditemukan.")
except KeyError:
	print(f"Error: Pastikan nama kolom di Excel ('Bobot (%)' dan '{kolom_lokasi}') sudah benar.")