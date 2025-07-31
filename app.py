import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import requests
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import folium
from folium.plugins import MarkerCluster
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# Optional: Ollama integration for AI capabilities
try:
	import ollama
	OLLAMA_AVAILABLE = True
except ImportError:
	OLLAMA_AVAILABLE = False
	print("Ollama not available. Some AI features will be limited.")

class SmartBusinessIntelligence:
	def __init__(self, data_path=None):
		"""Initialize the business intelligence system"""
		self.business_data = None
		self.competitor_data = None
		self.customer_data = None
		self.product_data = None
		self.geografis_data = None
		self.sales_data = None
		
		# Load data if provided
		if data_path:
			self.load_data(data_path)
			
		# Initialize NLTK for sentiment analysis
		try:
			nltk.download('vader_lexicon', quiet=True)
			self.sentiment_analyzer = SentimentIntensityAnalyzer()
		except:
			print("Warning: NLTK components couldn't be downloaded. Sentiment analysis will be limited.")
	
	def load_data(self, data_path):
		"""Load data from Excel or CSV file"""
		try:
			# Check file extension
			if data_path.endswith('.xlsx'):
				# Read Excel file with multiple sheets
				try:
					business_info = pd.read_excel(data_path, sheet_name='Business Info', header=3)
					product_data = pd.read_excel(data_path, sheet_name='Product', header=3)
					geografis_data = pd.read_excel(data_path, sheet_name='Info Geografis', header=3)
					
					# Print data for debugging
					print(f"Loaded business data: {len(business_info)} rows")
					print(f"Loaded product data: {len(product_data)} rows")
					print("Product data columns:", product_data.columns.tolist())
					
					# Clean and process data
					self.business_data = business_info.dropna(how='all').reset_index(drop=True)
					self.product_data = product_data.dropna(how='all').reset_index(drop=True)
					self.geografis_data = geografis_data.dropna(how='all').reset_index(drop=True)
					
					# Try to load customer data, but don't fail if it doesn't exist
					try:
						customer_data = pd.read_excel(data_path, sheet_name='Customer')
						self.customer_data = customer_data.dropna(how='all').reset_index(drop=True)
						print(f"Loaded customer data: {len(self.customer_data)} rows")
					except:
						print("No customer data sheet found")
						self.customer_data = None
					
					print(f"Successfully loaded data from {data_path}")
					
				except Exception as e:
					print(f"Error reading Excel sheets: {str(e)}")
					raise
					
			elif data_path.endswith('.csv'):
				# Read CSV file (assuming it's competitor data)
				self.competitor_data = pd.read_csv(data_path)
				print(f"Successfully loaded competitor data: {len(self.competitor_data)} records")
				
			else:
				print(f"Unsupported file format: {data_path}")
				
		except Exception as e:
			print(f"Error loading data: {str(e)}")
			# Initialize empty dataframes to prevent further errors
			self.business_data = pd.DataFrame()
			self.product_data = pd.DataFrame()
			self.customer_data = None
	
	def load_json_input(self, json_input):
		"""Load data from JSON input"""
		try:
			data = json.loads(json_input) if isinstance(json_input, str) else json_input
			
			# print("==== DATA INPUT ===")
			# print(data)
			# print("==== DATA INPUT ===")
			# Extract business data
			if 'business_data' in data:
				business = data['business_data']
				self.business_data = pd.DataFrame([{
					'Nama Usaha': business.get('business_name', ''),
					'Jenis Usaha': business.get('business_type', ''),
					'Latitude': business.get('location', {}).get('latitude', 0),
					'Longitude': business.get('location', {}).get('longitude', 0),
					'contact': '',  # Default empty values for missing fields
					'Rating': 0,
					'District': '',
					'City': ''
				}])
			
			# Extract menu/product data
			if 'product' in data.get('business_data', {}):
				menu_items = data['business_data']['product']
				self.product_data = pd.DataFrame([{
					'Nama Menu': item.get('item_name', ''),
					'Harga': float(item.get('price', 0)),
				} for item in menu_items])
			
			# print("=== DATA CUSTOMER LOAD JSON INPUT ===")		
			# print(data['customer_data'])		
			# print("=== DATA CUSTOMER LOAD JSON INPUT ===")	
			# Extract customer data if available
			if 'customer_data' in data:
				customers = data['customer_data']
				self.customer_data = pd.DataFrame([{
					'Nama Pelanggan': customer.get('name', ''),
					'Usia': customer.get('age', 0),
					'Jenis Kelamin': customer.get('gender', ''),
					'Frekuensi Kunjungan': customer.get('visit_frequency', ''),
					'Barang Favorit': customer.get('favorite_item', ''),
					'Tanggal Kunjungan Terakhir': customer.get('last_visit_date', '')
				} for customer in customers])
			
			# Extract sales data if available
			if 'historical_sales_data' in data:
				# Process historical sales data
				sales_records = []
				for sale in data['historical_sales_data']:
					for item in sale.get('items_purchased', []):
						sales_records.append({
							'date': sale.get('date', ''),
							'time': sale.get('time', ''),
							'customer_id': sale.get('customer_id', ''),
							'item_name': item.get('item_name', ''),
							'quantity': item.get('quantity', 0)
						})
				self.sales_data = pd.DataFrame(sales_records)
			
			print("Successfully loaded data from JSON input")
			
		except Exception as e:
			print(f"Error processing JSON input: {str(e)}")
	
	def load_competitor_data(self, competitor_file):
		"""Load competitor data from CSV file"""
		try:
			self.competitor_data = pd.read_csv(competitor_file)
			print(f"Successfully loaded competitor data: {len(self.competitor_data)} records")
		except Exception as e:
			print(f"Error loading competitor data: {str(e)}")

	def geografis_analysis(self):
		try:			
			# Pastikan total bobot adalah 1.0 (atau 100%)
			total_bobot = self.geografis_data['Bobot (%)'].sum()
			if not (0.99 <= total_bobot <= 1.01):
				print(f"Peringatan: Total bobot Anda adalah {total_bobot*100:.0f}%, bukan 100%. Harap periksa kembali.")
			else:
				# Hitung nilai dengan mengalikan bobot dan skor
				self.geografis_data['Nilai'] = self.geografis_data['Bobot (%)'] * self.geografis_data['Penilaian Lokasi (1-5)']

				# Hitung skor akhir dengan menjumlahkan semua nilai
				skor_akhir = self.geografis_data['Nilai'].sum()

				# Tampilkan hasil
				print("\n===== HASIL ANALISIS LOKASI USAHA =====")
				print(f"Lokasi: {'Penilaian Lokasi (1-5)'}")
				print(f"Skor Akhir: {skor_akhir:.2f} dari 5.00")
				print("=" * 40)

				# Berikan interpretasi sederhana
				if skor_akhir >= 4.0:
					return "📈 Interpretasi: Skor sangat baik. Lokasi ini memiliki potensi yang sangat kuat."
				elif skor_akhir >= 3.0:
					return "👍 Interpretasi: Skor baik. Lokasi ini memiliki potensi yang solid."
				elif skor_akhir >= 2.0:
					return "🤔 Interpretasi: Skor cukup. Ada beberapa aspek yang perlu diwaspadai atau diperbaiki."
				else:
					return "⚠️ Interpretasi: Skor rendah. Lokasi ini memiliki banyak tantangan, pertimbangkan dengan sangat hati-hati."

		except FileNotFoundError:
			print(f"Error: Sheet 'Geografis Info' tidak ditemukan.")
		except KeyError:
			print(f"Error: Pastikan nama kolom di Excel ('Bobot (%)' dan '{'Penilaian Lokasi (1-5)'}') sudah benar.")

	def cross_sell_intelligence(self):
		"""Generate product pairing and bundling recommendations"""

		# print("=== PRODUCT DATA ===")
		# print(self.product_data)
		# print(self.product_data['Nama Menu'])		
		# print("=== PRODUCT DATA ===")

		if self.product_data is None or len(self.product_data) < 2:
			return {"error": "Insufficient product data for cross-sell analysis",
					"product_pairings": [],
					"recommended_bundles": [],
					"time_based_recommendations": {}}
		
		# Baris ini sekarang akan berjalan tanpa error
		products = self.product_data['Nama Menu'].unique()
		
		# If we have sales data, use it to create better recommendations
		if hasattr(self, 'sales_data') and self.sales_data is not None:
			# Analyze which products are frequently bought together
			# This is a simplified version - in a real system, we'd use association rule mining
			print("Generating cross-sell recommendations based on sales data...")
			
			# For demonstration, we'll create random pairings
			import random
			pairings = []
			for product in products:
				# Select 2-3 random products to pair with this one
				num_pairs = random.randint(2, min(3, len(products)-1))
				potential_pairs = [p for p in products if p != product]
				pairs = random.sample(potential_pairs, num_pairs)
				
				for pair in pairs:
					confidence = round(random.uniform(0.3, 0.9), 2)
					pairings.append({
						"product": product,
						"recommended_with": pair,
						"confidence": confidence
					})
		else:
			# Without sales data, create recommendations based on price points and categories
			print("Generating cross-sell recommendations based on product attributes...")
			
			# Add category if it exists in the data
			has_categories = 'Kategori' in self.product_data.columns
			
			# Create a feature matrix for products
			product_features = self.product_data[['Nama Menu', 'Harga']].copy()
			if has_categories:
				# One-hot encode categories
				categories = pd.get_dummies(self.product_data['Kategori'], prefix='cat')
				product_features = pd.concat([product_features, categories], axis=1)
			
			# Normalize prices
			scaler = StandardScaler()
			price_scaled = scaler.fit_transform(product_features[['Harga']])
			product_features['price_scaled'] = price_scaled
			
			# Calculate similarity between products
			product_matrix = product_features.drop(['Nama Menu', 'Harga'], axis=1)
			similarity = cosine_similarity(product_matrix)
			
			# Create product pairings based on similarity
			pairings = []
			for i, product in enumerate(products):
				# Get top 3 most similar products
				similar_indices = similarity[i].argsort()[-4:-1][::-1]  # Exclude the product itself
				for idx in similar_indices:
					if idx < len(products) and products[idx] != product:
						# Calculate a confidence score based on similarity
						conf = round((similarity[i][idx] + 1) / 2, 2)  # Scale from [-1,1] to [0,1]
						pairings.append({
							"product": product,
							"recommended_with": products[idx],
							"confidence": conf
						})
		
		# Create bundle recommendations
		bundles = []
		
		# Group high-confidence pairings into bundles
		high_conf_pairings = [p for p in pairings if p['confidence'] > 0.6]
		
		# Create bundles of 2-3 products
		if high_conf_pairings:
			import random
			num_bundles = min(5, len(high_conf_pairings) // 2)
			
			for i in range(num_bundles):
				# Select a random pairing to start the bundle
				start_pair = random.choice(high_conf_pairings)
				bundle_products = [start_pair['product'], start_pair['recommended_with']]
				
				# Possibly add a third product
				if random.random() > 0.5:
					# Find products that pair well with either of the first two
					potential_thirds = [p['recommended_with'] for p in pairings 
									   if p['product'] in bundle_products 
									   and p['recommended_with'] not in bundle_products]
					
					if potential_thirds:
						bundle_products.append(random.choice(potential_thirds))
				
				# Calculate bundle price (original and discounted)
				original_price = sum(self.product_data[self.product_data['Nama Menu'].isin(bundle_products)]['Harga'])
				discount = round(random.uniform(0.1, 0.2), 2)  # 10-20% discount
				bundle_price = round(original_price * (1 - discount))
				
				bundles.append({
					"name": f"Bundle {i+1}: " + " + ".join(bundle_products),
					"products": bundle_products,
					"original_price": original_price,
					"bundle_price": bundle_price,
					"savings": round(original_price - bundle_price),
					"discount_percentage": f"{int(discount*100)}%"
				})
		
		# If we have Ollama available, generate creative bundle names
		if OLLAMA_AVAILABLE and bundles:
			try:
				for i, bundle in enumerate(bundles):
					prompt = f"Create a catchy, short marketing name (max 5 words) for a food bundle that includes: {', '.join(bundle['products'])}. Just return the name, nothing else."
					response = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
					bundle_name = response['message']['content'].strip().strip('"\'')
					bundles[i]["name"] = bundle_name
			except Exception as e:
				print(f"Error generating bundle names with Ollama: {str(e)}")
				
		return {
			"product_pairings": sorted(pairings, key=lambda x: x['confidence'], reverse=True),
			"recommended_bundles": bundles,
			"time_based_recommendations": self._generate_time_based_recommendations()
		}
	
	def _generate_time_based_recommendations(self):
		"""Generate time-based recommendations (morning, afternoon, evening)"""
		# This is a simplified version - in a real system, we'd analyze actual sales patterns by time
		
		# Get current time to make contextual recommendations
		current_hour = datetime.datetime.now().hour
		
		time_recommendations = {
			"morning": [],
			"afternoon": [],
			"evening": []
		}
		
		if self.product_data is not None:
			# Simple rules for demonstration
			for _, product in self.product_data.iterrows():
				product_name = product['Nama Menu']
				
				# Morning recommendations (coffee, breakfast items)
				if any(keyword in product_name.lower() for keyword in 
					  ['kopi', 'coffee', 'teh', 'tea', 'hot', 'panas', 'breakfast']):
					time_recommendations["morning"].append(product_name)
				
				# Afternoon recommendations (lunch, refreshing drinks)
				if any(keyword in product_name.lower() for keyword in 
					  ['es', 'iced', 'cold', 'dingin', 'lunch', 'makan siang']):
					time_recommendations["afternoon"].append(product_name)
				
				# Evening recommendations (dinner, desserts)
				if any(keyword in product_name.lower() for keyword in 
					  ['dinner', 'makan malam', 'dessert', 'large']):
					time_recommendations["evening"].append(product_name)
			
			# Limit each category to max 5 items
			for time_period in time_recommendations:
				if len(time_recommendations[time_period]) > 5:
					time_recommendations[time_period] = time_recommendations[time_period][:5]
		
		# Add a current time recommendation
		if 5 <= current_hour < 11:
			current_period = "morning"
		elif 11 <= current_hour < 17:
			current_period = "afternoon"
		else:
			current_period = "evening"
			
		return {
			"time_periods": time_recommendations,
			"current_recommendation": {
				"period": current_period,
				"items": time_recommendations[current_period]
			}
		}

	def phr_engine(self):
		"""Generate personalized recommendations for customers"""
		if self.customer_data is None or len(self.customer_data) == 0:
			return {"error": "No customer data available for personalization"}
		
		# Analyze customer segments
		customer_segments = self._analyze_customer_segments()
		
		# Generate personalized recommendations for each customer
		personalized_recommendations = []

		# print("=== DATA CUSTOMER PHR ENGINE===")		
		# print(self.customer_data)		
		# print("=== DATA CUSTOMER PHR ENGINE===")	

		# self.customer_data.columns = self.customer_data.iloc[0]
		# self.customer_data = self.customer_data[1:].reset_index(drop=True)	
		customer_phr = self.customer_data.copy()

		# print("=== CUSTOMER FEATURE ===")
		customer_phr.columns = customer_phr.iloc[0]  # Set header
		customer_phr = customer_phr[1:].reset_index(drop=True)
		# print(customer_phr)
		# print("=== CUSTOMER FEATURE ===")
		for _, customer in customer_phr.iterrows():
			customer_name = customer['Nama Pelanggan']
			favorite_item = customer.get('Barang Favorit', '')
			age = customer.get('Usia', 0)
			gender = customer.get('Jenis Kelamin', '')
			frequency = customer.get('Frekuensi Kunjungan', '')
			
			# Find the customer's segment
			segment = "Unknown"
			for seg in customer_segments:
				if customer_name in customer_segments[seg]['customers']:
					segment = seg
					break
			
			# Generate recommendations based on favorite item and segment
			recommendations = []
			
			if favorite_item and favorite_item in self.product_data['Nama Menu'].values:
				# Get similar products to favorite item
				similar_products = self._get_similar_products(favorite_item)
				recommendations.extend(similar_products[:3])
			
			# Add segment-based recommendations
			if segment != "Unknown" and 'recommended_products' in customer_segments[segment]:
				segment_recommendations = [p for p in customer_segments[segment]['recommended_products'] 
										  if p not in recommendations]
				recommendations.extend(segment_recommendations[:2])
			
			# Fill up to 5 recommendations if needed
			if len(recommendations) < 5 and len(self.product_data) > 0:
				remaining_products = [p for p in self.product_data['Nama Menu'] 
									 if p not in recommendations and p != favorite_item]
				import random
				additional = random.sample(remaining_products, min(5-len(recommendations), len(remaining_products)))
				recommendations.extend(additional)
			
			# Create personalization object
			personalization = {
				"customer_name": customer_name,
				"segment": segment,
				"favorite_item": favorite_item,
				"recommended_products": recommendations,
				"personalization_factors": {
					"age": int(age) if not pd.isna(age) else None,
					"gender": gender,
					"visit_frequency": frequency
				}
			}
			
			personalized_recommendations.append(personalization)
		
		return {
			"customer_segments": customer_segments,
			"personalized_recommendations": personalized_recommendations
		}
	
	def _analyze_customer_segments(self):
		
		# print("=== DATA CUSTOMER ANALYZE CUSTOMER SEGMENT===")		
		# print(self.customer_data)		
		# print("=== DATA CUSTOMER ANALYZE CUSTOMER SEGMENT===")	

		"""Segment customers based on demographics and behavior"""
		if len(self.customer_data) < 3:  # Need at least a few customers to segment
			# Create basic segments
			return {
				"Regular Customers": {
					"description": "Customers who visit frequently",
					"customers": self.customer_data[self.customer_data['Frekuensi Kunjungan'] == 'Sering']['Nama Pelanggan'].tolist(),
					"recommended_products": self._get_popular_products(5)
				},
				"Occasional Customers": {
					"description": "Customers who visit occasionally",
					"customers": self.customer_data[self.customer_data['Frekuensi Kunjungan'] != 'Sering']['Nama Pelanggan'].tolist(),
					"recommended_products": self._get_popular_products(5)
				}
			}
		
		# For more customers, do more sophisticated segmentation
		# Convert frequency to numeric
		frequency_map = {'Jarang': 1, 'Sedang': 2, 'Sering': 3}
		# self.customer_data = self.customer_data[1:].reset_index(drop=True)
		customer_features = self.customer_data.copy()

		# print("=== CUSTOMER FEATURE ===")
		customer_features.columns = customer_features.iloc[0]
		# print(customer_features)
		# print("=== CUSTOMER FEATURE ===")
		customer_features['Frequency_Numeric'] = customer_features['Frekuensi Kunjungan'].map(frequency_map)
		
		# Convert gender to numeric
		gender_map = {'Laki-laki': 0, 'Perempuan': 1}
		customer_features['Gender_Numeric'] = customer_features['Jenis Kelamin'].map(gender_map)
		
		# Select features for clustering
		features = ['Usia', 'Frequency_Numeric', 'Gender_Numeric']
		X = customer_features[features].dropna()
		
		if len(X) < 3:  # Not enough data for clustering
			return {
				"All Customers": {
					"description": "All customers",
					"customers": self.customer_data['Nama Pelanggan'].tolist(),
					"recommended_products": self._get_popular_products(5)
				}
			}
		
		# Normalize features
		scaler = StandardScaler()
		X_scaled = scaler.fit_transform(X)
		
		# Determine optimal number of clusters (simplified)
		n_clusters = min(3, len(X) // 2)  # At most 3 clusters, at least 2 customers per cluster
		
		# Apply KMeans clustering
		kmeans = KMeans(n_clusters=n_clusters, random_state=42)
		customer_features.loc[X.index, 'Cluster'] = kmeans.fit_predict(X_scaled)
		
		# Create segment descriptions
		segments = {}
		for cluster_id in range(n_clusters):
			cluster_customers = customer_features[customer_features['Cluster'] == cluster_id]
			
			# Determine segment characteristics
			avg_age = cluster_customers['Usia'].mean()
			common_frequency = cluster_customers['Frekuensi Kunjungan'].mode()[0]
			common_gender = cluster_customers['Jenis Kelamin'].mode()[0]
			
			# Create segment name and description
			if common_frequency == 'Sering':
				segment_name = f"Loyal {common_gender}"
			elif common_frequency == 'Sedang':
				segment_name = f"Regular {common_gender}"
			else:
				segment_name = f"Occasional {common_gender}"
				
			if avg_age < 30:
				segment_name += " (Young)"
			elif avg_age < 45:
				segment_name += " (Middle-aged)"
			else:
				segment_name += " (Mature)"
			
			# Find popular products for this segment
			segment_favorites = []
			for _, customer in cluster_customers.iterrows():
				if not pd.isna(customer['Barang Favorit']):
					segment_favorites.append(customer['Barang Favorit'])
			
			# Add more recommendations if needed
			recommended_products = segment_favorites[:3]
			if len(recommended_products) < 5:
				additional = [p for p in self._get_popular_products(5) if p not in recommended_products]
				recommended_products.extend(additional[:5-len(recommended_products)])
			
			segments[segment_name] = {
				"description": f"{common_gender} customers around age {int(avg_age)} who visit {common_frequency.lower()}",
				"customers": cluster_customers['Nama Pelanggan'].tolist(),
				"avg_age": round(avg_age, 1),
				"common_frequency": common_frequency,
				"common_gender": common_gender,
				"favorite_products": segment_favorites,
				"recommended_products": recommended_products
			}
		
		return segments
	
	def _get_similar_products(self, product_name):
		"""Find products similar to the given product"""
		if product_name not in self.product_data['Nama Menu'].values:
			return []
		
		# Get product features
		product_features = self.product_data.copy()
		
		# Extract product categories if available
		has_categories = 'Kategori' in product_features.columns
		if has_categories:
			categories = pd.get_dummies(product_features['Kategori'], prefix='cat')
			product_features = pd.concat([product_features, categories], axis=1)
		
		# Normalize prices
		scaler = StandardScaler()
		price_scaled = scaler.fit_transform(product_features[['Harga']])
		product_features['price_scaled'] = price_scaled
		
		# Calculate similarity between products
		feature_cols = ['price_scaled']
		if has_categories:
			feature_cols.extend([col for col in product_features.columns if col.startswith('cat_')])
			
		product_matrix = product_features[feature_cols].values
		similarity = cosine_similarity(product_matrix)
		
		# Find the index of the target product
		product_idx = product_features[product_features['Nama Menu'] == product_name].index[0]
		
		# Get indices of most similar products (excluding the product itself)
		similar_indices = similarity[product_idx].argsort()[::-1][1:6]  # Top 5 similar products
		
		# Return the names of similar products
		similar_products = product_features.iloc[similar_indices]['Nama Menu'].tolist()
		
		return similar_products
	
	def _get_popular_products(self, n=5):
		"""Get the most popular products based on available data"""
		if hasattr(self, 'sales_data') and self.sales_data is not None:
			# Use actual sales data if available
			popular = self.sales_data['item_name'].value_counts().head(n).index.tolist()
			return popular
		elif self.customer_data is not None and 'Barang Favorit' in self.customer_data.columns:
			# Use customer favorites
			popular = self.customer_data['Barang Favorit'].value_counts().head(n).index.tolist()
			return popular
		else:
			# Just return some products
			return self.product_data['Nama Menu'].head(n).tolist()

	def competitor_intelligence(self):
		"""Analyze competitors and market position"""
		if self.competitor_data is None:
			return {"error": "No competitor data available for analysis"}
		
		if self.business_data is None or len(self.business_data) == 0:
			return {"error": "Business data is required for competitor analysis"}
		
		# Get business location
		try:
			business_lat = float(self.business_data['Latitude'].iloc[0])
			business_lng = float(self.business_data['Longitude'].iloc[0])
		except:
			business_lat = 0
			business_lng = 0
			print("Warning: Business location not available. Using default coordinates.")
		
		# Filter relevant competitors (same category/nearby)
		relevant_competitors = self._filter_relevant_competitors(business_lat, business_lng)
		
		if len(relevant_competitors) == 0:
			return {"error": "No relevant competitors found for analysis"}
		
		# Analyze competitor ratings and reviews
		rating_analysis = self._analyze_competitor_ratings(relevant_competitors)
		
		# Analyze competitor locations
		location_analysis = self._analyze_competitor_locations(relevant_competitors, business_lat, business_lng)
		
		# Analyze competitor business hours
		hours_analysis = self._analyze_competitor_hours(relevant_competitors)
		
		# Analyze competitor pricing
		pricing_analysis = self._analyze_competitor_pricing(relevant_competitors)
		
		# Generate competitive positioning
		positioning = self._generate_competitive_positioning(
			relevant_competitors, rating_analysis, location_analysis, hours_analysis, pricing_analysis
		)
		
		return {
			"competitor_summary": {
				"total_competitors": len(relevant_competitors),
				"avg_rating": rating_analysis["avg_rating"],
				"top_competitors": rating_analysis["top_competitors"],
				"nearby_competitors": location_analysis["nearby_count"]
			},
			"rating_analysis": rating_analysis,
			"location_analysis": location_analysis,
			"hours_analysis": hours_analysis,
			"pricing_analysis": pricing_analysis,
			"competitive_positioning": positioning
		}
	
	def _filter_relevant_competitors(self, business_lat, business_lng):
		"""Filter competitors by relevance to the business"""
		
		# print("=== DATA BUSINESS FILTER RELEVANT COMPETITOR ===")
		# print(self.business_data)
		# print("=== DATA BUSINESS FILTER RELEVANT COMPETITOR ===")
		# self.business_data.columns = self.business_data.iloc[0]

		# self.business_data = self.business_data[1:].reset_index(drop=True)
		# Get business type
		business_type = self.business_data['Jenis Usaha'].iloc[0] if len(self.business_data) > 0 else ""
		
		# Filter by category relevance
		if 'categories' in self.competitor_data.columns:
			# Look for similar categories
			relevant_keywords = []
			if 'Kafe' in business_type or 'Cafe' in business_type:
				relevant_keywords = ['Kafe', 'Cafe', 'Coffee', 'Kopi']
			elif 'Restoran' in business_type or 'Restaurant' in business_type:
				relevant_keywords = ['Restoran', 'Restaurant', 'Rumah Makan', 'Warung']
			
			# Filter competitors by category
			if relevant_keywords:
				category_filter = self.competitor_data['categories'].apply(
					lambda x: any(keyword in str(x) for keyword in relevant_keywords)
				)
				relevant_competitors = self.competitor_data[category_filter].copy()
			else:
				relevant_competitors = self.competitor_data.copy()
		else:
			relevant_competitors = self.competitor_data.copy()
		
		# Calculate distance to each competitor
		if business_lat != 0 and business_lng != 0 and 'lat' in relevant_competitors.columns and 'lng' in relevant_competitors.columns:
			from math import radians, sin, cos, sqrt, atan2
			
			def haversine(lat1, lon1, lat2, lon2):
				# Calculate distance between two points in km
				R = 6371  # Earth radius in km
				dLat = radians(lat2 - lat1)
				dLon = radians(lon2 - lon1)
				a = sin(dLat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dLon/2)**2
				c = 2 * atan2(sqrt(a), sqrt(1-a))
				return R * c
			
			relevant_competitors['distance'] = relevant_competitors.apply(
				lambda row: haversine(business_lat, business_lng, row['lat'], row['lng']), 
				axis=1
			)
			
			# Sort by distance
			relevant_competitors = relevant_competitors.sort_values('distance')
			
			# Limit to competitors within 5km
			relevant_competitors = relevant_competitors[relevant_competitors['distance'] <= 5]
		
		# Limit to top 20 most relevant competitors
		if len(relevant_competitors) > 20:
			if 'distance' in relevant_competitors.columns:
				relevant_competitors = relevant_competitors.head(20)
			elif 'rating' in relevant_competitors.columns:
				relevant_competitors = relevant_competitors.sort_values('rating', ascending=False).head(20)
		
		return relevant_competitors
	
	def _analyze_competitor_ratings(self, competitors):
		"""Analyze competitor ratings and reviews"""
		if 'rating' not in competitors.columns:
			return {
				"avg_rating": None,
				"rating_distribution": {},
				"top_competitors": [],
				"bottom_competitors": []
			}
		
		# Calculate average rating
		avg_rating = competitors['rating'].mean()
		
		# Create rating distribution
		rating_distribution = {}
		for i in range(1, 6):
			lower = i - 0.5
			upper = i + 0.5
			count = len(competitors[(competitors['rating'] >= lower) & (competitors['rating'] < upper)])
			rating_distribution[str(i)] = count
		
		# Get top and bottom competitors by rating
		top_competitors = competitors.sort_values('rating', ascending=False).head(5)
		bottom_competitors = competitors.sort_values('rating').head(5)
		
		top_list = []
		for _, comp in top_competitors.iterrows():
			top_list.append({
				"name": comp['name'],
				"rating": comp['rating'],
				"reviews": comp.get('reviews', 'N/A')
			})
		
		bottom_list = []
		for _, comp in bottom_competitors.iterrows():
			bottom_list.append({
				"name": comp['name'],
				"rating": comp['rating'],
				"reviews": comp.get('reviews', 'N/A')
			})
		
		return {
			"avg_rating": round(avg_rating, 2),
			"rating_distribution": rating_distribution,
			"top_competitors": top_list,
			"bottom_competitors": bottom_list
		}
	
	def _analyze_competitor_locations(self, competitors, business_lat, business_lng):
		"""Analyze competitor locations relative to the business"""
		if 'lat' not in competitors.columns or 'lng' not in competitors.columns:
			return {
				"nearby_count": 0,
				"distance_distribution": {},
				"closest_competitors": []
			}
		
		# Count competitors by distance
		distance_distribution = {
			"< 0.5 km": 0,
			"0.5 - 1 km": 0,
			"1 - 2 km": 0,
			"2 - 5 km": 0,
			"> 5 km": 0
		}
		
		for _, comp in competitors.iterrows():
			distance = comp.get('distance', 0)
			if distance < 0.5:
				distance_distribution["< 0.5 km"] += 1
			elif distance < 1:
				distance_distribution["0.5 - 1 km"] += 1
			elif distance < 2:
				distance_distribution["1 - 2 km"] += 1
			elif distance < 5:
				distance_distribution["2 - 5 km"] += 1
			else:
				distance_distribution["> 5 km"] += 1
		
		# Count nearby competitors (within 1km)
		nearby_count = distance_distribution["< 0.5 km"] + distance_distribution["0.5 - 1 km"]
		
		# Get closest competitors
		closest_competitors = []
		if 'distance' in competitors.columns:
			for _, comp in competitors.sort_values('distance').head(5).iterrows():
				closest_competitors.append({
					"name": comp['name'],
					"distance": round(comp['distance'], 2),
					"address": comp.get('address', 'N/A')
				})
		
		return {
			"nearby_count": nearby_count,
			"distance_distribution": distance_distribution,
			"closest_competitors": closest_competitors
		}
	
	def _analyze_competitor_hours(self, competitors):
		"""Analyze competitor business hours"""
		if 'operation_days' not in competitors.columns or 'open_time' not in competitors.columns:
			return {
				"hours_coverage": {},
				"early_openers": [],
				"late_closers": []
			}
		
		# Analyze operation days
		days_distribution = {
			"Everyday": 0,
			"Weekdays only": 0,
			"Weekends only": 0,
			"Other": 0
		}
		
		for _, comp in competitors.iterrows():
			op_days = str(comp.get('operation_days', ''))
			if op_days == 'Everyday':
				days_distribution["Everyday"] += 1
			elif op_days in ['Monday-Friday', 'Weekdays']:
				days_distribution["Weekdays only"] += 1
			elif op_days in ['Saturday-Sunday', 'Weekends']:
				days_distribution["Weekends only"] += 1
			else:
				days_distribution["Other"] += 1
		
		# Analyze opening hours
		early_openers = []
		late_closers = []
		
		for _, comp in competitors.iterrows():
			open_time = comp.get('open_time', '')
			close_time = comp.get('close_time', '')
			
			if open_time and str(open_time) < '08:00:00':
				early_openers.append({
					"name": comp['name'],
					"open_time": open_time,
					"close_time": close_time
				})
			
			if close_time and str(close_time) > '21:00:00':
				late_closers.append({
					"name": comp['name'],
					"open_time": open_time,
					"close_time": close_time
				})
		
		# Limit to top 5
		early_openers = early_openers[:5]
		late_closers = late_closers[:5]
		
		return {
			"days_distribution": days_distribution,
			"early_openers": early_openers,
			"late_closers": late_closers
		}
	
	def _analyze_competitor_pricing(self, competitors):
		"""Analyze competitor pricing"""
		price_ranges = []
		
		# Extract price ranges
		for _, comp in competitors.iterrows():
			min_price = comp.get('spend_price_min')
			max_price = comp.get('spend_price_max')
			
			if pd.notna(min_price) and pd.notna(max_price):
				price_ranges.append({
					"name": comp['name'],
					"min_price": min_price,
					"max_price": max_price,
					"avg_price": (min_price + max_price) / 2
				})
		
		# Calculate average price points
		if price_ranges:
			avg_min = sum(p['min_price'] for p in price_ranges) / len(price_ranges)
			avg_max = sum(p['max_price'] for p in price_ranges) / len(price_ranges)
			avg_price = sum(p['avg_price'] for p in price_ranges) / len(price_ranges)
		else:
			avg_min = avg_max = avg_price = 0
		
		# Compare with our prices
		our_prices = []
		if self.product_data is not None:
			our_min = self.product_data['Harga'].min()
			our_max = self.product_data['Harga'].max()
			our_avg = self.product_data['Harga'].mean()
			
			our_prices = {
				"min_price": our_min,
				"max_price": our_max,
				"avg_price": our_avg
			}
		
		return {
			"competitor_price_ranges": price_ranges,
			"market_average": {
				"min_price": round(avg_min),
				"max_price": round(avg_max),
				"avg_price": round(avg_price)
			},
			"our_prices": our_prices
		}
	
	def _generate_competitive_positioning(self, competitors, rating_analysis, location_analysis, 
										 hours_analysis, pricing_analysis):
		"""Generate competitive positioning insights"""
		strengths = []
		weaknesses = []
		opportunities = []
		threats = []

		# self.business_data.columns = self.business_data.iloc[0]
		# self.business_data = self.business_data[1:].reset_index(drop=True)
		# print("=== DATA BUSINESS GENERATE COMPETITIVE POSITIONING ===")
		# print(self.business_data)
		# print("=== DATA BUSINESS GENERATE COMPETITIVE POSITIONING ===")

		self.business_data['Rating'] = self.business_data['Rating'].astype(str).str.replace(',', '.').astype(float)

		# Analyze ratings
		if self.business_data is not None and 'Rating' in self.business_data.columns:
			our_rating = self.business_data['Rating'].iloc[0]
			market_avg = rating_analysis.get('avg_rating', 0)
			
			if our_rating > market_avg:
				strengths.append(f"Higher rating than market average ({our_rating} vs {market_avg})")
			else:
				weaknesses.append(f"Lower rating than market average ({our_rating} vs {market_avg})")
		
		# Analyze location
		nearby_count = location_analysis.get('nearby_count', 0)
		if nearby_count > 5:
			threats.append(f"High competition density with {nearby_count} competitors within 1km")
		elif nearby_count < 2:
			opportunities.append("Low competition density in immediate area")
		
		# Analyze pricing
		our_prices = pricing_analysis.get('our_prices', {})
		market_avg = pricing_analysis.get('market_average', {})
		
		if our_prices and market_avg:
			if our_prices.get('avg_price', 0) < market_avg.get('avg_price', 0):
				strengths.append("Lower average prices than competitors")
			else:
				weaknesses.append("Higher average prices than competitors")
		
		# Generate recommendations
		recommendations = []
		
		if len(strengths) > 0:
			recommendations.append(f"Leverage your strengths: {strengths[0]}")
		
		if len(weaknesses) > 0:
			recommendations.append(f"Address your main weakness: {weaknesses[0]}")
		
		if len(opportunities) > 0:
			recommendations.append(f"Explore this opportunity: {opportunities[0]}")
		
		if len(threats) > 0:
			recommendations.append(f"Mitigate this threat: {threats[0]}")
		
		# Add specific recommendations based on analysis
		if our_prices and market_avg and our_prices.get('avg_price', 0) > market_avg.get('avg_price', 0) * 1.2:
			recommendations.append("Consider introducing some lower-priced menu items to attract price-sensitive customers")
		
		if hours_analysis.get('early_openers') and len(hours_analysis['early_openers']) > 3:
			recommendations.append("Consider earlier opening hours to match competitors and capture morning traffic")
		
		if hours_analysis.get('late_closers') and len(hours_analysis['late_closers']) > 3:
			recommendations.append("Evaluate extending closing hours to match competitors and capture evening customers")
		
		return {
			"swot_analysis": {
				"strengths": strengths,
				"weaknesses": weaknesses,
				"opportunities": opportunities,
				"threats": threats
			},
			"recommendations": recommendations
		}

	def market_trend_observatory(self):
		"""Analyze market trends and consumer preferences"""
		# For a real system, this would connect to news APIs, social media APIs, etc.
		# Here we'll generate simulated trends based on our data
		
		# Generate food trends based on our product data
		food_trends = self._generate_food_trends()
		
		# Generate consumer preference trends
		consumer_trends = self._generate_consumer_trends()
		
		# Generate pricing trends
		pricing_trends = self._generate_pricing_trends()
		
		# Generate recommendations based on trends
		trend_recommendations = self._generate_trend_recommendations(
			food_trends, consumer_trends, pricing_trends
		)
		
		return {
			"food_trends": food_trends,
			"consumer_trends": consumer_trends,
			"pricing_trends": pricing_trends,
			"recommendations": trend_recommendations
		}
	
	def _generate_food_trends(self):
		"""Generate food trend analysis"""
		# In a real system, this would analyze news articles, social media, etc.
		# Here we'll generate simulated trends based on our data
		
		trends = []
		
		# Analyze our menu for potential trends
		if self.product_data is not None:
			product_names = self.product_data['Nama Menu'].tolist()
			
			# Look for trends in our menu
			trend_keywords = {
				"Coffee": ["kopi", "coffee", "espresso", "latte", "americano"],
				"Milk alternatives": ["oat", "almond", "soy"],
				"Local flavors": ["aren", "gula aren", "susu", "madu", "tiger"],
				"Matcha": ["matcha", "green tea"],
				"Cold drinks": ["cold", "es", "iced"],
				"Specialty drinks": ["signature", "special"]
			}
			
			# Count occurrences of each trend
			trend_counts = {}
			for trend, keywords in trend_keywords.items():
				count = sum(1 for product in product_names if any(keyword.lower() in product.lower() for keyword in keywords))
				if count > 0:
					trend_counts[trend] = count
			
			# Sort trends by count
			sorted_trends = sorted(trend_counts.items(), key=lambda x: x[1], reverse=True)
			
			# Generate trend insights
			for trend, count in sorted_trends:
				# Calculate percentage of menu
				percentage = round((count / len(product_names)) * 100)
				
				# Generate trend insight
				if percentage > 30:
					strength = "dominant"
				elif percentage > 15:
					strength = "significant"
				else:
					strength = "emerging"
				
				trends.append({
					"trend": trend,
					"strength": strength,
					"menu_percentage": percentage,
					"item_count": count,
					"example_items": [p for p in product_names if any(k.lower() in p.lower() for k in trend_keywords[trend])][:3]
				})
		
		# Add some simulated market trends if we have few or no trends
		if len(trends) < 3:
			simulated_trends = [
				{
					"trend": "Plant-based alternatives",
					"strength": "growing",
					"description": "Increasing demand for plant-based milk alternatives and menu items",
					"market_growth": "25% year-over-year"
				},
				{
					"trend": "Local sourcing",
					"strength": "strong",
					"description": "Customers increasingly value locally sourced ingredients and products",
					"market_growth": "15% year-over-year"
				},
				{
					"trend": "Health-conscious options",
					"strength": "dominant",
					"description": "Growing demand for lower-sugar, healthier beverage and food options",
					"market_growth": "30% year-over-year"
				}
			]
			
			# Add simulated trends that don't overlap with our detected trends
			existing_trend_names = [t["trend"] for t in trends]
			for trend in simulated_trends:
				if trend["trend"] not in existing_trend_names:
					trends.append(trend)
		
		return trends
	
	def _generate_consumer_trends(self):
		"""Generate consumer preference trend analysis"""
		# In a real system, this would analyze customer data, reviews, etc.
		# Here we'll generate simulated consumer trends
		
		consumer_trends = [
			{
				"trend": "Convenience and speed",
				"strength": "strong",
				"description": "Customers increasingly value quick service and convenient ordering options",
				"recommendation": "Consider implementing online ordering or a mobile app"
			},
			{
				"trend": "Personalization",
				"strength": "growing",
				"description": "Customers expect personalized recommendations and customizable options",
				"recommendation": "Offer customization options for popular menu items"
			},
			{
				"trend": "Experience-focused consumption",
				"strength": "emerging",
				"description": "Customers seek unique experiences, not just products",
				"recommendation": "Create Instagram-worthy presentation and ambiance"
			}
		]
		
		return consumer_trends
	
	def _generate_pricing_trends(self):
		"""Generate pricing trend analysis"""
		# In a real system, this would analyze historical pricing data, competitor pricing, etc.
		# Here we'll generate simulated pricing trends
		
		pricing_trends = []
		
		# Analyze our pricing if available
		if self.product_data is not None and 'Harga' in self.product_data.columns:
			# Calculate price ranges
			min_price = self.product_data['Harga'].min()
			max_price = self.product_data['Harga'].max()
			avg_price = self.product_data['Harga'].mean()
			
			# Generate price distribution
			price_ranges = {
				"< 20k": len(self.product_data[self.product_data['Harga'] < 20000]),
				"20k - 30k": len(self.product_data[(self.product_data['Harga'] >= 20000) & (self.product_data['Harga'] < 30000)]),
				"30k - 40k": len(self.product_data[(self.product_data['Harga'] >= 30000) & (self.product_data['Harga'] < 40000)]),
				"> 40k": len(self.product_data[self.product_data['Harga'] >= 40000])
			}
			
			pricing_trends.append({
				"trend": "Price distribution",
				"min_price": int(min_price),
				"max_price": int(max_price),
				"avg_price": int(avg_price),
				"distribution": price_ranges
			})
		
		# Add simulated market pricing trends
		pricing_trends.extend([
			{
				"trend": "Premium pricing for specialty items",
				"strength": "growing",
				"description": "Customers willing to pay premium prices for unique, high-quality specialty items",
				"recommendation": "Consider adding premium items to your menu"
			},
			{
				"trend": "Value bundling",
				"strength": "strong",
				"description": "Increasing popularity of value bundles and combo deals",
				"recommendation": "Create strategic bundles to increase average order value"
			}
		])
		
		return pricing_trends
	
	def _generate_trend_recommendations(self, food_trends, consumer_trends, pricing_trends):
		"""Generate recommendations based on trend analysis"""
		recommendations = []
		
		# Add recommendations based on food trends
		for trend in food_trends:
			if trend.get("strength") in ["dominant", "strong", "growing"]:
				recommendations.append({
					"category": "Menu Development",
					"based_on": f"{trend.get('trend')} trend",
					"recommendation": f"Expand your {trend.get('trend')} offerings to capitalize on this strong trend"
				})
			elif trend.get("strength") in ["emerging", "growing"]:
				recommendations.append({
					"category": "Menu Testing",
					"based_on": f"{trend.get('trend')} trend",
					"recommendation": f"Test limited-time offerings featuring {trend.get('trend')} to gauge customer interest"
				})
		
		# Add recommendations based on consumer trends
		for trend in consumer_trends:
			if "recommendation" in trend:
				recommendations.append({
					"category": "Customer Experience",
					"based_on": f"{trend.get('trend')} trend",
					"recommendation": trend.get("recommendation")
				})
		
		# Add recommendations based on pricing trends
		for trend in pricing_trends:
			if "recommendation" in trend:
				recommendations.append({
					"category": "Pricing Strategy",
					"based_on": f"{trend.get('trend')} trend",
					"recommendation": trend.get("recommendation")
				})
		
		# Limit to top 5 most relevant recommendations
		recommendations = recommendations[:5]
		
		return recommendations
	def generate_dashboard(self):
		"""Generate a comprehensive business intelligence dashboard"""
		# Collect data from all modules
		cross_sell_data = self.cross_sell_intelligence()
		phr_data = self.phr_engine()
		competitor_data = self.competitor_intelligence()
		trend_data = self.market_trend_observatory()
		#geografis_data = self.geografis_analysis()
		
		# Combine into a comprehensive dashboard
		dashboard = {
			"business_summary": self._generate_business_summary(),
			"product_insights": self._generate_product_insights(),
			"customer_insights": self._generate_customer_insights(),
			"market_position": self._generate_market_position(competitor_data),
			"geografis_analysis" : self._generate_geografis_analysis(),
			"trend_analysis": self._generate_trend_analysis(trend_data),
			"recommendations": self._generate_consolidated_recommendations(
				cross_sell_data, phr_data, competitor_data, trend_data
			)
		}
		
		return dashboard
	
	def _generate_geografis_analysis(self):
		"""Generate insights about geographic/location analysis"""
		try:
			if self.geografis_data is None or len(self.geografis_data) == 0:
				return {
					"status": "no_data",
					"score": None,
					"interpretation": "Tidak ada data geografis yang tersedia."
				}
			total_bobot = self.geografis_data['Bobot (%)'].sum()
			if not (0.99 <= total_bobot <= 1.01):
				return {
					"status": "warning",
					"score": None,
					"interpretation": f"Peringatan: Total bobot Anda adalah {total_bobot*100:.0f}%, bukan 100%. Harap periksa kembali."
				}
			# Hitung nilai dengan mengalikan bobot dan skor
			self.geografis_data['Nilai'] = self.geografis_data['Bobot (%)'] * self.geografis_data['Penilaian Lokasi (1-5)']
			skor_akhir = self.geografis_data['Nilai'].sum()
			# Interpretasi
			if skor_akhir >= 4.0:
				interpretation = "📈 Skor sangat baik. Lokasi ini memiliki potensi yang sangat kuat."
			elif skor_akhir >= 3.0:
				interpretation = "👍 Skor baik. Lokasi ini memiliki potensi yang solid."
			elif skor_akhir >= 2.0:
				interpretation = "🤔 Skor cukup. Ada beberapa aspek yang perlu diperbaiki."
			else:
				interpretation = "⚠️ Skor rendah. Lokasi ini memiliki banyak tantangan."
			return {
				"status": "success",
				"score": round(skor_akhir, 2),
				"interpretation": interpretation,
				"detail": self.geografis_data[['Bobot (%)', 'Penilaian Lokasi (1-5)', 'Nilai']].to_dict(orient='records')
			}
		except Exception as e:
			return {
				"status": "error",
				"score": None,
				"interpretation": f"Error: {str(e)}"
			}

	def _generate_business_summary(self):
		"""Generate a summary of the business"""
		summary = {
			"name": "Unknown Business",
			"type": "Unknown",
			"location": {"lat": 0, "lng": 0},
			"product_count": 0,
			"avg_price": 0
		}
		
		if self.business_data is not None and len(self.business_data) > 0:
			summary["name"] = self.business_data['Nama Usaha'].iloc[0]
			summary["type"] = self.business_data['Jenis Usaha'].iloc[0]
			summary["location"] = {
				"lat": self.business_data['Latitude'].iloc[0],
				"lng": self.business_data['Longitude'].iloc[0]
			}
		
		if self.product_data is not None:
			summary["product_count"] = len(self.product_data)
			if 'Harga' in self.product_data.columns:
				summary["avg_price"] = int(self.product_data['Harga'].mean())
		
		return summary
	
	def _generate_product_insights(self):
		"""Generate insights about products"""
		insights = {
			"total_products": 0,
			"price_range": {"min": 0, "max": 0, "avg": 0},
			"price_distribution": {},
			"top_products": []
		}
		
		if self.product_data is not None and len(self.product_data) > 0:
			insights["total_products"] = len(self.product_data)
			
			if 'Harga' in self.product_data.columns:
				insights["price_range"] = {
					"min": int(self.product_data['Harga'].min()),
					"max": int(self.product_data['Harga'].max()),
					"avg": int(self.product_data['Harga'].mean())
				}
				
				# Generate price distribution
				price_ranges = {
					"< 20k": len(self.product_data[self.product_data['Harga'] < 20000]),
					"20k - 30k": len(self.product_data[(self.product_data['Harga'] >= 20000) & (self.product_data['Harga'] < 30000)]),
					"30k - 40k": len(self.product_data[(self.product_data['Harga'] >= 30000) & (self.product_data['Harga'] < 40000)]),
					"> 40k": len(self.product_data[self.product_data['Harga'] >= 40000])
				}
				insights["price_distribution"] = price_ranges
			
			# Get top products (in a real system, this would be based on sales)
			top_products = []
			for _, product in self.product_data.head(5).iterrows():
				top_products.append({
					"name": product['Nama Menu'],
					"price": int(product['Harga'])
				})
			insights["top_products"] = top_products
		
		return insights
	
	def _generate_customer_insights(self):
		"""Generate insights about customers"""
		insights = {
			"total_customers": 0,
			"segments": {},
			"top_customers": []
		}
		
		if self.customer_data is not None and len(self.customer_data) > 0:
			# Bersihkan header jika perlu
			customer_df = self.customer_data.copy()
			customer_df.columns = customer_df.iloc[0]
			customer_df = customer_df[1:].reset_index(drop=True)

			insights["total_customers"] = len(customer_df)

			# Get customer segments from PHR engine
			phr_data = self.phr_engine()
			if "customer_segments" in phr_data:
				segments = {}
				for segment, data in phr_data["customer_segments"].items():
					segments[segment] = {
						"count": len(data.get("customers", [])),
						"description": data.get("description", ""),
						"favorite_products": data.get("favorite_products", [])[:3]
					}
				insights["segments"] = segments

			# Get top customers
			top_customers = []
			for _, customer in customer_df.head(5).iterrows():
				top_customers.append({
					"name": customer.get('Nama Pelanggan', ''),
					"frequency": customer.get('Frekuensi Kunjungan', ''),
					"favorite_item": customer.get('Barang Favorit', '')
				})
			insights["top_customers"] = top_customers
		
		return insights
	
	def _generate_market_position(self, competitor_data):
		"""Generate insights about market position"""
		if "error" in competitor_data:
			return {
				"status": "incomplete",
				"message": competitor_data["error"]
			}
		
		# Extract key competitor insights
		position = {
			"competitor_count": competitor_data.get("competitor_summary", {}).get("total_competitors", 0),
			"avg_market_rating": competitor_data.get("competitor_summary", {}).get("avg_rating", 0),
			"nearby_competitors": competitor_data.get("competitor_summary", {}).get("nearby_competitors", 0),
			"top_competitors": competitor_data.get("rating_analysis", {}).get("top_competitors", [])[:3],
			"pricing_position": "unknown"
		}
		
		# Determine pricing position
		our_prices = competitor_data.get("pricing_analysis", {}).get("our_prices", {})
		market_avg = competitor_data.get("pricing_analysis", {}).get("market_average", {})
		
		if our_prices and market_avg:
			our_avg = our_prices.get("avg_price", 0)
			market_avg_price = market_avg.get("avg_price", 0)
			
			if our_avg < market_avg_price * 0.8:
				position["pricing_position"] = "significantly below market"
			elif our_avg < market_avg_price * 0.95:
				position["pricing_position"] = "below market"
			elif our_avg <= market_avg_price * 1.05:
				position["pricing_position"] = "at market"
			elif our_avg <= market_avg_price * 1.2:
				position["pricing_position"] = "above market"
			else:
				position["pricing_position"] = "significantly above market"
		
		# Add SWOT analysis
		position["swot"] = competitor_data.get("competitive_positioning", {}).get("swot_analysis", {})
		
		return position
	
	def _generate_trend_analysis(self, trend_data):
		"""Generate insights about market trends"""
		# Extract key trends
		trends = {
			"food_trends": [t.get("trend") for t in trend_data.get("food_trends", [])],
			"consumer_trends": [t.get("trend") for t in trend_data.get("consumer_trends", [])],
			"pricing_trends": [t.get("trend") for t in trend_data.get("pricing_trends", [])]
		}
		
		# Add detailed trend information
		trends["top_food_trends"] = trend_data.get("food_trends", [])[:3]
		trends["top_consumer_trends"] = trend_data.get("consumer_trends", [])[:3]
		
		return trends
	
	def _generate_consolidated_recommendations(self, cross_sell_data, phr_data, competitor_data, trend_data):
		"""Generate consolidated recommendations from all modules"""
		all_recommendations = []
		
		# Add cross-sell recommendations
		if "recommended_bundles" in cross_sell_data:
			for bundle in cross_sell_data["recommended_bundles"][:2]:
				all_recommendations.append({
					"category": "Product Bundling",
					"recommendation": f"Create a bundle with {', '.join(bundle['products'])} at {bundle['bundle_price']} IDR ({bundle['discount_percentage']} discount)",
					"impact": "Increase average order value",
					"priority": "high"
				})
		
		# Add customer segment recommendations
		if "customer_segments" in phr_data:
			for segment, data in list(phr_data["customer_segments"].items())[:2]:
				if "recommended_products" in data:
					all_recommendations.append({
						"category": "Customer Targeting",
						"recommendation": f"Target {segment} segment with {', '.join(data['recommended_products'][:3])}",
						"impact": "Increase customer loyalty and frequency",
						"priority": "medium"
					})
		
		# Add competitor recommendations
		if "competitive_positioning" in competitor_data:
			for rec in competitor_data["competitive_positioning"].get("recommendations", [])[:2]:
				all_recommendations.append({
					"category": "Competitive Strategy",
					"recommendation": rec,
					"impact": "Improve competitive position",
					"priority": "high"
				})
		
		# Add trend recommendations
		if "recommendations" in trend_data:
			for rec in trend_data["recommendations"][:2]:
				all_recommendations.append({
					"category": rec.get("category", "Market Trend"),
					"recommendation": rec.get("recommendation", ""),
					"impact": "Capitalize on market trends",
					"priority": "medium"
				})
		
		# Sort by priority
		priority_order = {"high": 0, "medium": 1, "low": 2}
		all_recommendations.sort(key=lambda x: priority_order.get(x["priority"], 3))
		
		return all_recommendations
	
	def visualize_data(self):
		"""Generate visualizations of key insights"""
		# In a real implementation, this would generate actual visualizations
		# Here we'll just return the data that would be visualized
		
		visualizations = {
			"product_price_distribution": self._get_product_price_distribution(),
			"competitor_map": self._get_competitor_map_data(),
			"trend_analysis": self._get_trend_visualization_data(),
			"customer_segments": self._get_customer_segment_visualization()
		}
		
		return visualizations
	
	def _get_product_price_distribution(self):
		"""Get data for product price distribution visualization"""
		if self.product_data is None or 'Harga' not in self.product_data.columns:
			return {"status": "no_data"}
		
		# Create price ranges
		price_ranges = [
			{"range": "< 20k", "count": len(self.product_data[self.product_data['Harga'] < 20000])},
			{"range": "20k - 30k", "count": len(self.product_data[(self.product_data['Harga'] >= 20000) & (self.product_data['Harga'] < 30000)])},
			{"range": "30k - 40k", "count": len(self.product_data[(self.product_data['Harga'] >= 30000) & (self.product_data['Harga'] < 40000)])},
			{"range": "> 40k", "count": len(self.product_data[self.product_data['Harga'] >= 40000])}
		]
		
		return {
			"status": "success",
			"chart_type": "bar",
			"data": price_ranges,
			"title": "Product Price Distribution"
		}
	
	def _get_competitor_map_data(self):
		"""Get data for competitor map visualization"""
		if self.competitor_data is None or 'lat' not in self.competitor_data.columns:
			return {"status": "no_data"}
		
		# Get business location
		if self.business_data is not None and len(self.business_data) > 0:
			business_lat = self.business_data['Latitude'].iloc[0]
			business_lng = self.business_data['Longitude'].iloc[0]
			business_name = self.business_data['Nama Usaha'].iloc[0]
		else:
			business_lat = 0
			business_lng = 0
			business_name = "Your Business"
		
		# Get competitor locations
		competitors = []
		for _, comp in self.competitor_data.iterrows():
			if pd.notna(comp['lat']) and pd.notna(comp['lng']):
				competitors.append({
					"name": comp['name'],
					"lat": comp['lat'],
					"lng": comp['lng'],
					"rating": comp.get('rating', 'N/A')
				})
		
		return {
			"status": "success",
			"chart_type": "map",
			"business": {
				"name": business_name,
				"lat": business_lat,
				"lng": business_lng
			},
			"competitors": competitors,
			"title": "Competitor Map"
		}
	
	def _get_trend_visualization_data(self):
		"""Get data for trend visualization"""
		# Get trend data
		trend_data = self.market_trend_observatory()
		
		# Extract food trends
		food_trends = []
		for trend in trend_data.get("food_trends", []):
			strength_value = {
				"dominant": 5,
				"strong": 4,
				"significant": 3,
				"growing": 2,
				"emerging": 1
			}.get(trend.get("strength", ""), 0)
			
			food_trends.append({
				"trend": trend.get("trend", ""),
				"strength": strength_value,
				"strength_label": trend.get("strength", "")
			})
		
		return {
			"status": "success",
			"chart_type": "radar",
			"data": food_trends,
			"title": "Food Trend Strength Analysis"
		}
	
	def _get_customer_segment_visualization(self):
		"""Get data for customer segment visualization"""
		# Get customer segment data
		phr_data = self.phr_engine()
		
		if "customer_segments" not in phr_data:
			return {"status": "no_data"}
		
		# Extract segment sizes
		segments = []
		for segment, data in phr_data["customer_segments"].items():
			segments.append({
				"segment": segment,
				"count": len(data.get("customers", [])),
				"description": data.get("description", "")
			})
		
		return {
			"status": "success",
			"chart_type": "pie",
			"data": segments,
			"title": "Customer Segments"
		}
	
	def generate_report(self):
		"""Generate a comprehensive business intelligence report"""
		# Get data from all modules
		dashboard = self.generate_dashboard()
		visualizations = self.visualize_data()
		
		# Create report structure
		report = {
			"title": f"Business Intelligence Report for {dashboard['business_summary']['name']}",
			"date": datetime.datetime.now().strftime("%Y-%m-%d"),
			"executive_summary": self._generate_executive_summary(dashboard),
			"business_overview": dashboard["business_summary"],
			"product_analysis": {
				"insights": dashboard["product_insights"],
				"visualization": visualizations["product_price_distribution"]
			},
			"customer_analysis": {
				"insights": dashboard["customer_insights"],
				"visualization": visualizations["customer_segments"]
			},
			"competitor_analysis": {
				"insights": dashboard["market_position"],
				"visualization": visualizations["competitor_map"]
			},
			"geografis_analysis": {
				"insights": dashboard["geografis_analysis"],
				"visualization": visualizations["trend_analysis"]
			},
			"market_trends": {
				"insights": dashboard["trend_analysis"],
				"visualization": visualizations["trend_analysis"]
			},
			"strategic_recommendations": dashboard["recommendations"],
			"conclusion": self._generate_conclusion(dashboard)
		}
		
		return report
	
	def _generate_executive_summary(self, dashboard):
		"""Generate an executive summary for the report"""
		business_name = dashboard["business_summary"]["name"]
		product_count = dashboard["product_insights"]["total_products"]
		avg_price = dashboard["product_insights"]["price_range"]["avg"]
		competitor_count = dashboard["market_position"].get("competitor_count", 0)
		
		summary = f"This report provides a comprehensive analysis of {business_name}, "
		summary += f"a business with {product_count} products at an average price point of {avg_price} IDR. "
		
		if competitor_count > 0:
			summary += f"The business operates in a competitive environment with {competitor_count} identified competitors. "
		
		# Add key recommendations
		if dashboard["recommendations"]:
			top_rec = dashboard["recommendations"][0]
			summary += f"Our top recommendation is to {top_rec['recommendation'].lower()}."
		
		return summary
	
	def _generate_conclusion(self, dashboard):
		"""Generate a conclusion for the report"""
		business_name = dashboard["business_summary"]["name"]
		
		conclusion = f"Based on our comprehensive analysis, {business_name} has several opportunities "
		conclusion += "to strengthen its market position and drive growth. "
		
		# Add strengths
		strengths = dashboard["market_position"].get("swot", {}).get("strengths", [])
		if strengths:
			conclusion += f"The business should leverage its key strengths, particularly {strengths[0].lower()}. "
		
		# Add trend alignment
		food_trends = dashboard["trend_analysis"].get("food_trends", [])
		if food_trends:
			conclusion += f"Aligning with market trends such as {food_trends[0]} will be crucial for future success. "
		
		# Add final recommendation
		conclusion += "By implementing the strategic recommendations outlined in this report, "
		conclusion += f"{business_name} can enhance its competitive position and drive sustainable growth."
		
		return conclusion

	def run_simulation(self, scenario):
		"""Run a business scenario simulation"""
		if scenario == "price_adjustment":
			return self._simulate_price_adjustment()
		elif scenario == "new_product":
			return self._simulate_new_product()
		elif scenario == "competitor_entry":
			return self._simulate_competitor_entry()
		else:
			return {"error": f"Unknown scenario: {scenario}"}
	
	def _simulate_price_adjustment(self):
		"""Simulate the impact of price adjustments"""
		if self.product_data is None or 'Harga' not in self.product_data.columns:
			return {"error": "Product data with prices is required for this simulation"}
		
		# Calculate current metrics
		current_avg_price = self.product_data['Harga'].mean()
		
		# Simulate different price adjustment scenarios
		scenarios = []
		
		# Scenario 1: 10% price increase
		increase_scenario = {
			"name": "Price Increase (10%)",
			"avg_price": round(current_avg_price * 1.1),
			"estimated_impact": {
				"revenue": "+7%",
				"customer_volume": "-3%",
				"profit_margin": "+8%"
			},
			"recommendation": "Consider for premium products with inelastic demand"
		}
		scenarios.append(increase_scenario)
		
		# Scenario 2: 10% price decrease
		decrease_scenario = {
			"name": "Price Decrease (10%)",
			"avg_price": round(current_avg_price * 0.9),
			"estimated_impact": {
				"revenue": "+5%",
				"customer_volume": "+15%",
				"profit_margin": "-6%"
			},
			"recommendation": "Consider for products with elastic demand to drive volume"
		}
		scenarios.append(decrease_scenario)
		
		# Scenario 3: Tiered pricing
		tiered_scenario = {
			"name": "Tiered Pricing Strategy",
			"description": "Introduce economy, standard, and premium versions of popular products",
			"estimated_impact": {
				"revenue": "+12%",
				"customer_volume": "+8%",
				"profit_margin": "+4%"
			},
			"recommendation": "Implement for top 5 products to capture different customer segments"
		}
		scenarios.append(tiered_scenario)
		
		return {
			"current_avg_price": round(current_avg_price),
			"scenarios": scenarios,
			"recommended_scenario": "Tiered Pricing Strategy",
			"implementation_steps": [
				"Identify top 5 products by sales volume",
				"Create economy versions with 15% lower prices and reduced portions/ingredients",
				"Create premium versions with 20% higher prices and enhanced quality/presentation",
				"Implement for a 3-month trial period",
				"Measure impact on sales volume, revenue, and customer feedback"
			]
		}
	
	def _simulate_new_product(self):
		"""Simulate the impact of introducing new products"""
		# Analyze market trends
		trend_data = self.market_trend_observatory()
		
		# Identify top trends
		top_trends = [t.get("trend") for t in trend_data.get("food_trends", [])][:3]
		
		# Generate product ideas based on trends
		product_ideas = []
		
		if "Coffee" in top_trends or "Specialty drinks" in top_trends:
			product_ideas.append({
				"name": "Signature Cold Brew Series",
				"description": "A line of premium cold brew coffee with various flavor infusions",
				"price_point": "32,000 - 38,000 IDR",
				"target_segment": "Coffee enthusiasts and young professionals",
				"estimated_impact": {
					"revenue": "+8%",
					"new_customers": "+12%"
				}
			})
		
		if "Milk alternatives" in top_trends:
			product_ideas.append({
				"name": "Plant-Based Latte Collection",
				"description": "Coffee drinks made with oat, almond, and soy milk alternatives",
				"price_point": "35,000 - 42,000 IDR",
				"target_segment": "Health-conscious customers and those with dietary restrictions",
				"estimated_impact": {
					"revenue": "+6%",
					"new_customers": "+15%"
				}
			})
		
		if "Local flavors" in top_trends:
			product_ideas.append({
				"name": "Indonesian Heritage Series",
				"description": "Beverages featuring traditional Indonesian flavors like pandan, gula aren, and coconut",
				"price_point": "28,000 - 35,000 IDR",
				"target_segment": "Customers seeking authentic local experiences",
				"estimated_impact": {
					"revenue": "+10%",
					"new_customers": "+8%"
				}
			})
		
		# Add a generic idea if we don't have enough
		if len(product_ideas) < 3:
			product_ideas.append({
				"name": "Seasonal Limited Edition Drinks",
				"description": "Rotating menu of seasonal specialties to create urgency and excitement",
				"price_point": "30,000 - 40,000 IDR",
				"target_segment": "Variety seekers and trend followers",
				"estimated_impact": {
					"revenue": "+7%",
					"new_customers": "+9%"
				}
			})
		
		# Determine recommended idea
		recommended_idea = product_ideas[0]["name"] if product_ideas else "None"
		
		return {
			"market_trends": top_trends,
			"product_ideas": product_ideas,
			"recommended_idea": recommended_idea,
			"implementation_steps": [
				"Develop recipes and conduct internal taste tests",
				"Run a limited-time promotion to gauge customer interest",
				"Collect customer feedback and refine offerings",
				"Train staff on preparation and presentation",
				"Launch with targeted marketing to relevant customer segments"
			]
		}
	
	def _simulate_competitor_entry(self):
		"""Simulate the impact of a new competitor entering the market"""
		# Create a hypothetical new competitor
		new_competitor = {
			"name": "Trendy New Coffee Chain",
			"description": "A modern coffee chain with Instagram-worthy aesthetics and innovative menu",
			"strengths": [
				"Strong social media presence",
				"Unique signature drinks",
				"Modern, appealing ambiance"
			],
			"weaknesses": [
				"Higher price points",
				"Limited food menu",
				"New to the market with unproven customer loyalty"
			]
		}
		
		# Estimate potential impact
		impact = {
			"customer_volume": "-8% to -15% initially",
			"revenue": "-5% to -12% initially",
			"most_affected_products": ["Specialty coffee drinks", "Instagram-worthy beverages"]
		}
		
		# Generate response strategies
		strategies = [
			{
				"name": "Differentiation Strategy",
				"description": "Emphasize unique aspects of your business that the competitor cannot easily replicate",
				"actions": [
					"Highlight local sourcing and community connections",
					"Showcase signature items unique to your business",
					"Emphasize customer service and personal relationships"
				],
				"estimated_effectiveness": "High"
			},
			{
				"name": "Loyalty Program Enhancement",
				"description": "Strengthen customer loyalty to reduce switching",
				"actions": [
					"Introduce or enhance loyalty rewards program",
					"Create member-exclusive menu items or events",
					"Implement personalized offers based on purchase history"
				],
				"estimated_effectiveness": "Medium-High"
			},
			{
				"name": "Limited-Time Innovation",
				"description": "Create excitement with new offerings",
				"actions": [
					"Launch limited-time menu items to generate buzz",
					"Collaborate with local businesses for unique products",
					"Create 'secret menu' items promoted through social media"
				],
				"estimated_effectiveness": "Medium"
			}
		]
		
		return {
			"new_competitor": new_competitor,
			"potential_impact": impact,
			"response_strategies": strategies,
			"recommended_strategy": "Differentiation Strategy",
			"implementation_steps": [
				"Conduct customer surveys to identify your unique selling points",
				"Develop marketing materials highlighting your differentiators",
				"Train staff to emphasize your unique advantages",
				"Implement at least one signature item that cannot be easily copied",
				"Monitor competitor activities and customer feedback"
			]
		}

def main():
	"""Main function to demonstrate the system"""
	# Initialize the system
	sbi = SmartBusinessIntelligence()
	
	# Load merchant data
	sbi.load_data("Merchant Template.xlsx")
	
	# Load competitor data
	sbi.load_competitor_data("competitor.csv")

	geografis = sbi.geografis_analysis()
	print("\n=== Geografis Analysis ===")
	print(geografis)
	
	# Generate cross-sell recommendations
	cross_sell = sbi.cross_sell_intelligence()
	print("\n=== Cross-Sell Intelligence ===")
	if "error" in cross_sell:
		print(f"Error: {cross_sell['error']}")
	else:
		print(f"Generated {len(cross_sell.get('product_pairings', []))} product pairings")
		print(f"{cross_sell.get('product_pairings', [])}")
		print(f"\nRecommended {len(cross_sell.get('recommended_bundles', []))} product bundles")
		print(f"{cross_sell.get('recommended_bundles', [])}")
		
	# Generate personalized recommendations
	phr = sbi.phr_engine()
	print("\n=== Personalized Human Recommendation Engine ===")
	if "error" in phr:
		print(phr["error"])
	else:
		print(f"Identified {len(phr['customer_segments'])} customer segments")
		print(f"{phr['customer_segments']}")
		print(f"\nGenerated recommendations for {len(phr['personalized_recommendations'])} customers")
		print(f"{phr['personalized_recommendations']}")
	
	# Generate competitor intelligence
	comp_intel = sbi.competitor_intelligence()
	print("\n=== Competitor Intelligence Hub ===")
	if "error" in comp_intel:
		print(comp_intel["error"])
	else:
		print(f"Analyzed {comp_intel['competitor_summary']['total_competitors']} competitors")
		print(f"Average competitor rating: {comp_intel['competitor_summary']['avg_rating']}")
	
	# Generate market trend analysis
	trends = sbi.market_trend_observatory()
	print("\n=== Market Trend Observatory ===")
	print(f"Identified {len(trends['food_trends'])} food trends")
	print(f"{trends['food_trends']} food trends")
	print(f"\nIdentified {len(trends['consumer_trends'])} consumer trends")
	print(f"{trends['consumer_trends']} consumer trends")
	print(f"\nGenerated {len(trends['recommendations'])} trend-based recommendations")
	print(f"Generated {trends['recommendations']} trend-based recommendations")
	
	# Generate comprehensive report
	report = sbi.generate_report()
	print("\n=== Business Intelligence Report ===")
	print(f"Title: {report['title']}")
	print(f"Date: {report['date']}")
	print(f"Executive Summary: {report['executive_summary']}")
	print(f"Strategic Recommendations: {len(report['strategic_recommendations'])}")
	print(f"Strategic Explain: {report['strategic_recommendations']}")
	
	# Run a simulation
	simulation = sbi.run_simulation("price_adjustment")
	print("\n=== Price Adjustment Simulation ===")
	print(f"Current average price: {simulation['current_avg_price']} IDR")
	print(f"Recommended scenario: {simulation['recommended_scenario']}")
	
	print("\nSystem execution complete.")

if __name__ == "__main__":
	main()
