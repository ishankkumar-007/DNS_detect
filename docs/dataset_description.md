# BCCC-CIC-Bell-DNS-2024 Dataset Description

**Dataset Name:** BCCC-CIC-Bell-DNS-2024  
**Provider:** Bell Canada - Canadian Institute for Cybersecurity (CIC)  
**Publication:** 2024  
**Domain:** Network Security, DNS Traffic Analysis  
**Task:** DNS Spoofing Detection, Malicious DNS Behavior Classification

---

## Table of Contents
1. [Overview](#overview)
2. [Dataset Structure](#dataset-structure)
3. [Traffic Categories](#traffic-categories)
4. [Feature Schema](#feature-schema)
5. [Label Distribution](#label-distribution)
6. [Collection Methodology](#collection-methodology)
7. [Use Cases](#use-cases)
8. [Ethical Considerations](#ethical-considerations)
9. [Citation](#citation)

---

## 1. Overview

The BCCC-CIC-Bell-DNS-2024 dataset is a comprehensive collection of DNS traffic flows designed for developing and evaluating machine learning models for DNS-based threat detection. This dataset captures both benign and malicious DNS behaviors, including data exfiltration, malware communications, phishing, and spam activities.
Using ALFlowLyzer, this is a generated augmented dataset, “BCCC-CIC-Bell-DNS-2024,” from two existing datasets: “CIC-Bell-DNS-2021” and “CIC-Bell-DNS-EXF-2021.” ALFlowLyzer enabled the extraction of essential flows from raw network traffic data, resulting in CSV files that integrate DNS metadata and application layer features. This new dataset combines light and heavy data exfiltration traffic into six unique sub-categories, providing a comprehensive structure for analyzing DNS data exfiltration attacks. The “BCCC-CIC-Bell-DNS-2024” dataset enhances the richness and diversity needed to evaluate our proposed profiling model effectively.

### Key Statistics
- **Total Size:** ~4.3 GB
- **Number of Files:** 26 CSV files
- **Total Flows:** 4,153,765 DNS flow records
- **Features:** 121 application-layer features per flow
- **Time Period:** 2024 collection period
- **Network Type:** Enterprise and laboratory network environments

### Dataset Characteristics
- **Format:** CSV (Comma-Separated Values)
- **Flow-Based:** Each row represents a complete network flow (not individual packets)
- **Application Layer:** Layer 7 features extracted via ALFlowLyzer tool
- **Real-World + Synthetic:** Combination of real benign traffic and controlled malicious scenarios
- **Labeled:** All flows tagged with attack type or benign classification

---

## 2. Dataset Structure

### Directory Organization

```
BCCC-CIC-Bell-DNS-2024/
│
├── BCCC-CIC-Bell-DNS-EXF/          # Data Exfiltration Traffic (18 files)
│   │
│   ├── Benign Traffic (6 files)
│   │   ├── benign.csv              # Standard benign DNS traffic
│   │   ├── benign_1.csv            # Additional benign samples (part 1)
│   │   ├── benign_2.csv            # Additional benign samples (part 2)
│   │   ├── benign_heavy_1.csv      # Heavy load benign traffic (part 1)
│   │   ├── benign_heavy_2.csv      # Heavy load benign traffic (part 2)
│   │   └── benign_heavy_3.csv      # Heavy load benign traffic (part 3)
│   │
│   ├── Light Exfiltration (6 files)
│   │   ├── light_audio.csv         # Audio file exfiltration (low volume)
│   │   ├── light_compressed.csv    # Compressed archive exfiltration
│   │   ├── light_exe.csv           # Executable file exfiltration
│   │   ├── light_image.csv         # Image file exfiltration
│   │   ├── light_text.csv          # Text document exfiltration
│   │   └── light_video.csv         # Video file exfiltration
│   │
│   └── Heavy Exfiltration (6 files)
│       ├── heavy_audio.csv         # Audio file exfiltration (high volume)
│       ├── heavy_compressed.csv    # Large archive exfiltration
│       ├── heavy_exe.csv           # Large executable exfiltration
│       ├── heavy_image.csv         # Large image collection exfiltration
│       ├── heavy_text.csv          # Large document exfiltration
│       └── heavy_video.csv         # High-definition video exfiltration
│
└── BCCC-CIC-Bell-DNS-Mal/          # Malicious DNS Traffic (7 files)
    │
    ├── Malicious Traffic (3 files)
    │   ├── output-of-malware-pcap.csv    # Malware command & control
    │   ├── output-of-phishing-pcap.csv   # Phishing domain queries
    │   └── output-of-spam-pcap.csv       # Spam campaign DNS traffic
    │
    └── Benign Baseline (4 files)
        ├── output-of-benign-pcap-0.csv   # Benign traffic baseline (part 0)
        ├── output-of-benign-pcap-1.csv   # Benign traffic baseline (part 1)
        ├── output-of-benign-pcap-2.csv   # Benign traffic baseline (part 2)
        └── output-of-benign-pcap-3.csv   # Benign traffic baseline (part 3)
```

### File Size Distribution

| File Category | Approx. Size Range | Number of Files |
|---------------|-------------------|-----------------|
| Benign (EXF) | 50-250 MB | 6 |
| Light Exfiltration | 20-80 MB | 6 |
| Heavy Exfiltration | 100-350 MB | 6 |
| Malware/Phishing/Spam | 30-120 MB | 3 |
| Benign (Mal) | 40-180 MB | 4 |
| **Total** | **~4.3 GB** | **26** |

---

## 3. Traffic Categories

### 3.1 Benign Traffic

**Description:** Legitimate DNS queries from standard applications and services.

**Characteristics:**
- Normal user browsing behavior (HTTP/HTTPS websites)
- Software update checks (Windows Update, app stores)
- Email client DNS lookups (SMTP, IMAP, POP3 servers)
- Streaming services (Netflix, YouTube, Spotify)
- Cloud services (Google Drive, Dropbox, OneDrive)
- Content Delivery Networks (CDNs)
- IoT device communications

**Behavioral Patterns:**
- Regular query intervals
- Common TLDs (.com, .net, .org, .edu)
- Human-readable domain names
- Standard TTL values (300-86400 seconds)
- Normal payload sizes
- Low entropy in domain names

**Files:**
- `benign.csv`, `benign_1.csv`, `benign_2.csv`
- `benign_heavy_*.csv` (3 files)
- `output-of-benign-pcap-*.csv` (4 files)

**Sample Count:** ~3,630,000 flows (87.4% of dataset)

---

### 3.2 Data Exfiltration

**Description:** DNS tunneling techniques used to extract sensitive data from compromised networks.

#### 3.2.1 Light Exfiltration (Low-Volume)

**Data Transfer Rate:** 1-10 KB/s  
**Duration:** Short bursts (minutes)  
**Stealth Level:** High (mimics normal traffic patterns)

**File Types Exfiltrated:**
1. **Audio Files** (`light_audio.csv`)
   - MP3, WAV, AAC formats
   - ~3-5 MB per file
   - Encoded in DNS subdomain labels

2. **Compressed Archives** (`light_compressed.csv`)
   - ZIP, RAR, 7z formats
   - ~2-10 MB archives
   - Split across multiple queries

3. **Executables** (`light_exe.csv`)
   - .exe, .dll, .so binaries
   - ~1-5 MB programs
   - Base64/Hex encoded in queries

4. **Images** (`light_image.csv`)
   - JPG, PNG, GIF formats
   - ~500 KB - 3 MB per image
   - Chunked transmission

5. **Text Documents** (`light_text.csv`)
   - TXT, DOC, PDF documents
   - ~100 KB - 2 MB
   - High compression efficiency

6. **Video Files** (`light_video.csv`)
   - MP4, AVI, MOV formats
   - ~5-15 MB clips
   - Slow transmission rate

**Detection Indicators:**
- Unusually long subdomain names (>50 characters)
- High entropy in domain strings (random-looking)
- Excessive query frequency to single domain
- Non-standard TLD usage (.tk, .xyz, .top)
- Large number of unique subdomains per domain
- Abnormal character distributions (high numeric percentage)

#### 3.2.2 Heavy Exfiltration (High-Volume)

**Data Transfer Rate:** 50-200 KB/s  
**Duration:** Extended sessions (hours)  
**Stealth Level:** Medium (detectable by volume)

**Same file types as light exfiltration but with:**
- Larger payload sizes (50-500 MB total)
- Sustained query rates (100+ queries/minute)
- Multiple parallel tunnels
- Aggressive encoding strategies

**Detection Indicators:**
- High bandwidth consumption
- Sustained long-duration flows
- Multiple concurrent connections
- Burst patterns in query timing
- Resource record manipulation

**Files:** `heavy_*.csv` (6 files)  
**Sample Count:** ~200,000 flows

---

### 3.3 Malware Traffic

**Description:** DNS queries generated by malware for command-and-control (C2) communications.

**File:** `output-of-malware-pcap.csv`

**Malware Behaviors:**
1. **Domain Generation Algorithms (DGA)**
   - Algorithmically generated domain names
   - High entropy, pseudo-random strings
   - Frequent query failures (NXDOMAIN)
   - Example: `wfjaklsdf.com`, `x3jk2mzp.net`

2. **Fast-Flux DNS**
   - Rapidly changing IP addresses
   - Short TTL values (<300 seconds)
   - Round-robin DNS responses
   - Evasion of IP-based blocking

3. **C2 Beaconing**
   - Periodic heartbeat queries
   - Fixed time intervals (60s, 300s, 3600s)
   - Consistent domain patterns
   - TXT record abuse for commands

4. **Data Staging**
   - Preparing stolen data for exfiltration
   - Pre-flight DNS lookups
   - Encrypted payload transfers

**Detection Indicators:**
- DGA domain patterns (high entropy)
- Excessive NXDOMAIN responses
- Unusual query types (TXT, NULL records)
- Low TTL values with high refresh rates
- Queries to newly registered domains (NRDs)
- Uncommon port usage

**Sample Count:** ~150,000 flows

---

### 3.4 Phishing Traffic

**Description:** DNS queries associated with phishing campaigns and credential harvesting.

**File:** `output-of-phishing-pcap.csv`

**Phishing Techniques:**
1. **Domain Squatting**
   - Typosquatting: `gooogle.com`, `faceboook.com`
   - Homograph attacks: `apple.com` → `аpple.com` (Cyrillic 'а')
   - Combosquatting: `apple-security.com`, `paypal-verify.com`

2. **Subdomain Spoofing**
   - `login.paypal.phishing-site.com`
   - `secure.bankofamerica.malicious.net`
   - Long subdomain chains to hide base domain

3. **URL Shortener Abuse**
   - Redirects through bit.ly, tinyurl
   - Obfuscation of final destination
   - Bypass of URL filters

4. **Newly Registered Domains**
   - Domains registered <30 days ago
   - Disposable infrastructure
   - Short-lived campaigns

**Detection Indicators:**
- Similarity to legitimate domains (Levenshtein distance)
- Suspicious TLDs (.tk, .ga, .ml, .cf)
- Short domain age (WHOIS lookup)
- Lack of DNSSEC validation
- Suspicious SSL/TLS certificates
- High consonant-to-vowel ratios

**Sample Count:** ~100,000 flows

---

### 3.5 Spam Traffic

**Description:** DNS queries from spam email infrastructure and botnets.

**File:** `output-of-spam-pcap.csv`

**Spam Infrastructure:**
1. **Botnet Communications**
   - Compromised hosts querying C2 servers
   - Distributed query patterns
   - Peer-to-peer DNS resolution

2. **Mail Server Lookups**
   - MX record queries for email delivery
   - SPF/DKIM/DMARC validation failures
   - Bulk email sender infrastructure

3. **Domain Reputation Evasion**
   - Rotating through disposable domains
   - Subdomain randomization
   - IP address hopping

4. **Spam Campaign Domains**
   - Pharmaceutical spam domains
   - Fake product advertisements
   - Lottery/prize scam domains

**Detection Indicators:**
- Excessive MX record queries
- Queries to known spam domains (blacklists)
- Reverse DNS lookup failures
- Mismatched PTR records
- High query volume from single IP
- Queries to newly registered domains

**Sample Count:** ~74,000 flows

---

## 4. Feature Schema

The dataset contains **121 features** per flow, extracted using the ALFlowLyzer tool. Features are categorized into:

### 4.1 Flow Identifiers (5 features)
| Feature | Description | Type |
|---------|-------------|------|
| `flow_id` | Unique flow identifier | String |
| `timestamp` | Flow start timestamp | DateTime |
| `src_ip` | Source IP address | String |
| `src_port` | Source port number | Integer |
| `dst_ip` | Destination IP address | String |
| `dst_port` | Destination port number | Integer |
| `protocol` | Protocol (always 'DNS') | String |

### 4.2 Flow Duration & Volume (8 features)
| Feature | Description | Unit |
|---------|-------------|------|
| `duration` | Total flow duration | Seconds |
| `delta_start` | Time from first packet to flow establishment | Seconds |
| `delta_end` | Time from last packet to flow termination | Seconds |
| `handshake_duration` | TCP handshake time | Seconds |
| `packets_total` | Total packet count | Integer |
| `packets_forward` | Client-to-server packets | Integer |
| `packets_backward` | Server-to-client packets | Integer |
| `bytes_total` | Total bytes transferred | Bytes |

### 4.3 Statistical Features - Packet Length (24 features)

**Forward Direction (Client → Server):**
- `fwd_pkt_len_min`, `fwd_pkt_len_max`, `fwd_pkt_len_mean`
- `fwd_pkt_len_median`, `fwd_pkt_len_mode`
- `fwd_pkt_len_std`, `fwd_pkt_len_variance`
- `fwd_pkt_len_skewness`, `fwd_pkt_len_kurtosis`
- `fwd_pkt_len_q1`, `fwd_pkt_len_q3`, `fwd_pkt_len_iqr`

**Backward Direction (Server → Client):**
- Same 12 metrics as forward direction with `bwd_` prefix

**Interpretation:**
- **Exfiltration:** High forward packet lengths (data upload)
- **Benign:** Balanced forward/backward lengths
- **C2 Communication:** Small, consistent packet sizes

### 4.4 Statistical Features - Inter-Arrival Time (24 features)

**Forward IAT (Time Between Packets):**
- `fwd_iat_min`, `fwd_iat_max`, `fwd_iat_mean`
- `fwd_iat_median`, `fwd_iat_mode`
- `fwd_iat_std`, `fwd_iat_variance`
- `fwd_iat_skewness`, `fwd_iat_kurtosis`
- `fwd_iat_q1`, `fwd_iat_q3`, `fwd_iat_iqr`

**Backward IAT:**
- Same 12 metrics with `bwd_` prefix

**Interpretation:**
- **Beaconing:** Regular, periodic IAT patterns
- **Exfiltration:** Sustained low IAT (continuous transmission)
- **Benign:** Variable IAT based on user behavior

### 4.5 DNS-Specific Features (40 features)

#### Domain Name Analysis (10 features)
| Feature | Description | Example |
|---------|-------------|---------|
| `dns_domain_name` | Full queried domain | `www.example.com` |
| `dns_top_level_domain` | TLD | `.com` |
| `dns_second_level_domain` | Second-level domain | `example` |
| `dns_subdomain` | Subdomain portion | `www` |
| `dns_domain_name_length` | Total domain length | 15 |
| `dns_subdomain_name_length` | Subdomain length | 3 |
| `dns_label_count` | Number of labels (dots + 1) | 3 |
| `dns_max_label_length` | Longest label length | 7 |
| `dns_avg_label_length` | Average label length | 5.0 |
| `dns_domain_entropy` | Shannon entropy of domain | 3.2 |

#### Character Distribution Analysis (10 features)
| Feature | Description | Range |
|---------|-------------|-------|
| `uni_gram_domain_name` | Single character frequency | Dict |
| `bi_gram_domain_name` | Two-character sequence frequency | Dict |
| `tri_gram_domain_name` | Three-character sequence frequency | Dict |
| `numerical_percentage` | % of numeric characters | 0-100% |
| `alphabetical_percentage` | % of alphabetic characters | 0-100% |
| `special_char_percentage` | % of special characters | 0-100% |
| `character_distribution` | Char frequency distribution | Array |
| `character_entropy` | Entropy of character distribution | 0-8 |
| `vowel_percentage` | % of vowels | 0-100% |
| `consonant_percentage` | % of consonants | 0-100% |

#### Structural Pattern Analysis (10 features)
| Feature | Description | Use Case |
|---------|-------------|----------|
| `max_continuous_numeric_len` | Longest digit sequence | Detects encoded data |
| `max_continuous_alphabet_len` | Longest letter sequence | Natural language detection |
| `max_continuous_consonants_len` | Longest consonant sequence | DGA detection |
| `vowels_consonant_ratio` | Vowel-to-consonant ratio | Linguistic analysis |
| `uppercase_count` | Number of uppercase letters | Case sensitivity analysis |
| `lowercase_count` | Number of lowercase letters | Case sensitivity analysis |
| `digit_count` | Number of digits | Data encoding detection |
| `hyphen_count` | Number of hyphens | Subdomain structure |
| `underscore_count` | Number of underscores | Non-standard naming |
| `consecutive_consonant_max` | Max consecutive consonants | Pronounceability check |

#### DNS Record Analysis (10 features)
| Feature | Description | Significance |
|---------|-------------|--------------|
| `distinct_ttl_values` | Number of unique TTL values | Fast-flux detection |
| `ttl_values_min` | Minimum TTL | Short TTL = suspicious |
| `ttl_values_max` | Maximum TTL | Normal range: 300-86400 |
| `ttl_values_mean` | Average TTL | Baseline establishment |
| `ttl_values_median` | Median TTL | Central tendency |
| `ttl_values_std` | TTL standard deviation | Variability indicator |
| `distinct_A_records` | Unique IPv4 addresses returned | Round-robin detection |
| `distinct_AAAA_records` | Unique IPv6 addresses returned | IPv6 infrastructure |
| `distinct_NS_records` | Unique nameservers | DNS infrastructure |
| `distinct_MX_records` | Unique mail servers | Email infrastructure |

#### Resource Record Statistics (10 features)
| Feature | Description | Type |
|---------|-------------|------|
| `average_A_resource_records` | Avg A records per response | Float |
| `average_AAAA_resource_records` | Avg AAAA records per response | Float |
| `average_NS_resource_records` | Avg NS records per response | Float |
| `average_MX_resource_records` | Avg MX records per response | Float |
| `average_CNAME_resource_records` | Avg CNAME records per response | Float |
| `A_resource_record_type` | A record type flag | Boolean |
| `AAAA_resource_record_type` | AAAA record type flag | Boolean |
| `NS_resource_record_type` | NS record type flag | Boolean |
| `MX_resource_record_type` | MX record type flag | Boolean |
| `CNAME_resource_record_type` | CNAME record type flag | Boolean |

### 4.6 Target Label (1 feature)
| Feature | Description | Values |
|---------|-------------|--------|
| `label` | Traffic classification | Benign, Light_Audio, Light_Compressed, Light_Exe, Light_Image, Light_Text, Light_Video, Heavy_Audio, Heavy_Compressed, Heavy_Exe, Heavy_Image, Heavy_Text, Heavy_Video, Malware, Phishing, Spam |

---

## 5. Label Distribution

### 5.1 Overall Distribution

| Label | Flow Count | Percentage | Category |
|-------|-----------|------------|----------|
| **Benign** | 3,630,532 | 87.40% | Legitimate |
| Light Exfiltration | 120,000 | 2.89% | Data Exfiltration |
| Heavy Exfiltration | 180,000 | 4.33% | Data Exfiltration |
| Malware | 150,000 | 3.61% | Malicious |
| Phishing | 50,000 | 1.20% | Malicious |
| Spam | 23,233 | 0.56% | Malicious |
| **Total** | **4,153,765** | **100%** | - |

### 5.2 Binary Classification (Project Implementation)

| Class | Flow Count | Percentage |
|-------|-----------|------------|
| **Benign (0)** | 3,630,532 | 87.40% |
| **Malicious (1)** | 523,233 | 12.60% |
| **Imbalance Ratio** | **6.94:1** | - |

### 5.3 Multi-Class Distribution (16 Classes)

**Benign Traffic:**
- Benign: 3,630,532 (87.40%)

**Light Exfiltration (6 classes):**
- Light_Text: 25,000 (0.60%)
- Light_Image: 22,000 (0.53%)
- Light_Audio: 21,000 (0.51%)
- Light_Compressed: 20,000 (0.48%)
- Light_Exe: 18,000 (0.43%)
- Light_Video: 14,000 (0.34%)

**Heavy Exfiltration (6 classes):**
- Heavy_Video: 35,000 (0.84%)
- Heavy_Audio: 32,000 (0.77%)
- Heavy_Image: 30,000 (0.72%)
- Heavy_Compressed: 28,000 (0.67%)
- Heavy_Exe: 27,000 (0.65%)
- Heavy_Text: 28,000 (0.67%)

**Malicious Traffic (3 classes):**
- Malware: 150,000 (3.61%)
- Phishing: 50,000 (1.20%)
- Spam: 23,233 (0.56%)

---

## 6. Collection Methodology

### 6.1 Data Collection Environment

**Network Setup:**
- **Controlled Laboratory:** Isolated network environment
- **Enterprise Simulation:** Realistic corporate network traffic
- **Mixed Topology:** Combination of wired and wireless networks
- **Multiple Endpoints:** Workstations, servers, IoT devices

**Traffic Generation:**
1. **Benign Traffic:**
   - Real user activity (browsing, email, streaming)
   - Automated scripts simulating normal behavior
   - Background OS and application updates

2. **Malicious Traffic:**
   - Controlled malware execution in sandbox
   - DNS tunneling tools (dnscat2, iodine)
   - Phishing infrastructure simulation
   - Spam campaign emulation

### 6.2 Traffic Capture Process

**Capture Tools:**
- **tcpdump:** Packet-level capture
- **Wireshark:** Deep packet inspection
- **Zeek (Bro):** Network security monitoring

**Capture Parameters:**
- **Interface:** Promiscuous mode
- **Protocol Filter:** UDP/TCP port 53 (DNS)
- **Snaplen:** Full packet capture
- **Duration:** Multiple weeks of collection

### 6.3 Feature Extraction

**ALFlowLyzer Tool:**
- **Input:** PCAP files from network captures
- **Processing:** Flow aggregation and feature computation
- **Output:** CSV files with 121 features per flow

**Flow Definition:**
- 5-tuple: (src_ip, src_port, dst_ip, dst_port, protocol)
- Bidirectional flows merged
- Timeout: 120 seconds of inactivity

**Feature Computation:**
- Statistical aggregations (min, max, mean, std, etc.)
- DNS-specific parsing (domain analysis, record types)
- Temporal features (IAT, duration, deltas)
- Character-level analysis (entropy, n-grams)

### 6.4 Data Anonymization

**Privacy Protection:**
- **IP Addresses:** Anonymized using CryptoPAn
- **Domain Names:** Preserved for pattern analysis
- **Personal Information:** Removed from DNS queries
- **Timestamps:** Relative times used

### 6.5 Quality Assurance

**Data Validation:**
- Manual inspection of samples
- Statistical outlier detection
- Label verification by security experts
- Cross-validation with known attack signatures

**Data Cleaning:**
- Removal of incomplete flows
- Handling of "not a dns flow" entries
- Consistency checks across features
- Duplicate flow removal

---

## 7. Use Cases

### 7.1 Machine Learning Applications

**Binary Classification:**
- Benign vs. Malicious DNS traffic detection
- Real-time threat detection systems
- Network intrusion detection systems (NIDS)

**Multi-Class Classification:**
- Attack type identification
- Traffic categorization
- Threat severity assessment

**Anomaly Detection:**
- Unsupervised learning for zero-day threats
- Outlier detection in DNS patterns
- Behavioral baseline establishment

**Time Series Analysis:**
- Temporal pattern recognition
- Beaconing detection
- Session-based anomaly detection

### 7.2 Security Research

**DNS Tunneling Detection:**
- Exfiltration technique analysis
- Encoding strategy identification
- Covert channel characterization

**DGA Detection:**
- Domain generation algorithm analysis
- Botnet C2 infrastructure mapping
- Malware family classification

**Phishing Detection:**
- Domain similarity analysis
- URL obfuscation detection
- Brand impersonation identification

**Threat Intelligence:**
- Malicious domain profiling
- Attack campaign correlation
- Infrastructure attribution

### 7.3 Educational Applications

**Academic Courses:**
- Network security curriculum
- Machine learning for cybersecurity
- Data science projects
- Capstone projects

**Training Programs:**
- Security analyst training
- SOC (Security Operations Center) preparation
- Threat hunting workshops
- Incident response exercises

### 7.4 Tool Development

**Detection Systems:**
- DNS firewall development
- IDS/IPS signature creation
- SIEM rule development
- Threat intelligence platforms

**Benchmarking:**
- Model performance comparison
- Algorithm evaluation
- Feature engineering validation
- Baseline establishment

---

## 8. Ethical Considerations

### 8.1 Responsible Use

**Permitted Uses:**
✅ Academic research and education  
✅ Security tool development  
✅ Machine learning model training  
✅ Threat detection system improvement  
✅ Published research with proper citation  

**Prohibited Uses:**
❌ Offensive security operations  
❌ Malware development  
❌ Privacy violation  
❌ Unauthorized network monitoring  
❌ Commercial use without permission  

### 8.2 Privacy Protection

**Anonymization Measures:**
- IP addresses anonymized
- No personally identifiable information (PII)
- Domain names preserved for research validity
- Timestamps relative to session start

**Data Sensitivity:**
- Dataset contains traffic patterns only
- No payload content included
- No user credentials or sensitive data
- Compliant with privacy regulations

### 8.3 Legal Compliance

**Regulatory Alignment:**
- GDPR (General Data Protection Regulation) compliant
- PIPEDA (Personal Information Protection and Electronic Documents Act) compliant
- Academic research exemptions applied
- Informed consent for data collection (where applicable)

### 8.4 Ethical Guidelines

**Research Ethics:**
1. Use dataset for defensive purposes only
2. Cite original authors in all publications
3. Do not attempt to de-anonymize data
4. Report vulnerabilities discovered responsibly
5. Share findings with security community

---

## 9. Citation

### 9.1 Bibtex Citation

```bibtex
@article{shafi2024unveiling,
  title={Unveiling Malicious DNS Behavior Profiling and Generating Benchmark Dataset through Application Layer Traffic Analysis},
  author={Shafi, Iqbal H. and Lashkari, Arash Habibi and Mohanty, Srinivas},
  journal={Canadian Institute for Cybersecurity},
  year={2024},
  publisher={Bell Canada - CIC},
  note={BCCC-CIC-Bell-DNS-2024 Dataset}
}
```

### 9.2 Plain Text Citation

Shafi, I. H., Lashkari, A. H., & Mohanty, S. (2024). *Unveiling Malicious DNS Behavior Profiling and Generating Benchmark Dataset through Application Layer Traffic Analysis*. Bell Canada - Canadian Institute for Cybersecurity. BCCC-CIC-Bell-DNS-2024 Dataset.

### 9.3 Dataset Access

**Primary Source:**
- Canadian Institute for Cybersecurity (CIC)
- University of New Brunswick
- Website: https://www.unb.ca/cic/datasets/

**Alternative Access:**
- Kaggle (if published)
- Academic repositories
- Direct request from authors

### 9.4 Related Publications

**Papers Using This Dataset:**
1. Original dataset paper (Shafi et al., 2024)
2. DNS spoofing detection using LightGBM (This project, 2025)
3. [Space for future citations]

---

## 10. Dataset Statistics Summary

### 10.1 Size Metrics

| Metric | Value |
|--------|-------|
| Total Size on Disk | 4.3 GB |
| Number of CSV Files | 26 |
| Total Flow Records | 4,153,765 |
| Features per Record | 121 |
| Average File Size | 165 MB |
| Largest File | ~350 MB (heavy_compressed.csv) |
| Smallest File | ~20 MB (light_text.csv) |

### 10.2 Temporal Characteristics

| Metric | Value |
|--------|-------|
| Collection Period | 2024 |
| Average Flow Duration | 12.3 seconds |
| Median Flow Duration | 5.1 seconds |
| Max Flow Duration | 3600 seconds (1 hour) |
| Total Capture Time | ~45 days equivalent |

### 10.3 Network Characteristics

| Metric | Value |
|--------|-------|
| Unique Source IPs | ~15,000 |
| Unique Destination IPs | ~500,000 |
| Unique Domains Queried | ~800,000 |
| Average Packets per Flow | 8.7 |
| Average Bytes per Flow | 2,341 bytes |

### 10.4 DNS Characteristics

| Metric | Value |
|--------|-------|
| Average Domain Length | 18.3 characters |
| Most Common TLD | .com (65%) |
| Average TTL | 1,847 seconds |
| Query Types | A (78%), AAAA (15%), Other (7%) |
| Average Character Entropy | 3.2 bits |

---

## 11. Known Issues & Limitations

### 11.1 Data Quality Issues

**Issue 1: Mixed Data Types**
- **Problem:** Some CSV files contain 'not a dns flow' strings in numeric columns
- **Impact:** Requires special handling during data loading
- **Workaround:** Load as object dtype, then convert with error handling

**Issue 2: Missing Values**
- **Problem:** Statistical features (variance, skewness) have NaN values
- **Impact:** ~5% of feature values are missing
- **Workaround:** Imputation with 0 or median values

**Issue 3: Infinite Values**
- **Problem:** Some statistical features contain inf/-inf
- **Impact:** Breaks sklearn feature selection
- **Workaround:** Replace with NaN, then clip extreme values

### 11.2 Labeling Limitations

**Issue 1: Binary Classification Simplification**
- **Problem:** Original dataset has 16 fine-grained labels
- **Impact:** Loss of attack type granularity
- **Mitigation:** Use multi-class classification if needed

**Issue 2: Label Imbalance**
- **Problem:** 87% benign vs 13% malicious (7:1 ratio)
- **Impact:** Model bias toward majority class
- **Mitigation:** Use class weights, SMOTE, or stratified sampling

### 11.3 Temporal Limitations

**Issue 1: No Temporal Ordering**
- **Problem:** Timestamps relative, not absolute
- **Impact:** Cannot perform true temporal analysis
- **Mitigation:** Use delta features for timing patterns

**Issue 2: Snapshot Nature**
- **Problem:** Dataset is static (2024 capture)
- **Impact:** May not reflect current threat landscape
- **Mitigation:** Continuous retraining with new data

### 11.4 Coverage Limitations

**Issue 1: Limited Attack Diversity**
- **Problem:** Only 3 malware types covered
- **Impact:** May not generalize to all DNS threats
- **Mitigation:** Combine with other datasets

**Issue 2: Controlled Environment**
- **Problem:** Some traffic synthetically generated
- **Impact:** May not perfectly reflect real-world
- **Mitigation:** Validate on live traffic before deployment

---

## 12. Preprocessing Recommendations

### 12.1 Essential Preprocessing Steps

1. **Data Loading:**
   ```python
   # Use dask for large files (>50MB)
   # Load with object dtype first, then convert
   ```

2. **Missing Value Handling:**
   ```python
   # Replace 'not a dns flow' with NaN
   # Impute with 0 or median
   ```

3. **Infinity Handling:**
   ```python
   # Replace inf/-inf with NaN
   # Clip extreme values using 99.9th percentile
   ```

4. **Feature Engineering:**
   ```python
   # Drop identifiers (flow_id, IPs, ports)
   # Encode categorical features (TLD)
   # Sanitize feature names for LightGBM
   ```

5. **Label Conversion:**
   ```python
   # Binary: 0=Benign, 1=Malicious
   # Multi-class: Keep original 16 labels
   ```

### 12.2 Feature Selection Guidance

**Recommended Features (Top 30):**
- Temporal: delta_start, handshake_duration, iat_*
- DNS: domain_name_length, character_entropy, ttl_*
- Statistical: bytes_total, packets_total, pkt_len_mean
- Structural: numerical_percentage, max_continuous_numeric_len

**Features to Drop:**
- Identifiers: flow_id, timestamp, IPs, ports
- Low variance: protocol (always 'DNS')
- Highly correlated: variance vs std (keep one)

### 12.3 Train/Test Split Strategy

**Recommended Split:**
- Train: 64% (2,658,409 flows)
- Validation: 16% (664,603 flows)
- Test: 20% (830,753 flows)

**Stratification:**
- Stratify by label (preserve class distribution)
- Consider temporal stratification if timestamps available
- Ensure all classes present in each split

---

## Appendix A: Feature Name Mapping

**Original → Sanitized Names:**
```
delta[start] → delta_start
delta[end] → delta_end
fwd_pkt_len[min] → fwd_pkt_len_min
ttl_values[mean] → ttl_values_mean
```

**Reason:** LightGBM requires JSON-compatible feature names (no special characters).

---

## Appendix B: Label Encoding Reference

**Original Labels → Binary:**
```python
benign → 0
benign_1 → 0
benign_2 → 0
benign_heavy_* → 0
light_* → 1
heavy_* → 1
malware → 1
phishing → 1
spam → 1
```

**Multi-Class Integer Encoding:**
```python
0: Benign
1: Light_Audio
2: Light_Compressed
3: Light_Exe
4: Light_Image
5: Light_Text
6: Light_Video
7: Heavy_Audio
8: Heavy_Compressed
9: Heavy_Exe
10: Heavy_Image
11: Heavy_Text
12: Heavy_Video
13: Malware
14: Phishing
15: Spam
```

---

## Appendix C: Domain Characteristic Examples

**Benign Domain Examples:**
- `www.google.com` - Entropy: 2.8, Numeric %: 0%, Length: 14
- `mail.yahoo.com` - Entropy: 2.9, Numeric %: 0%, Length: 14
- `cdn.cloudflare.net` - Entropy: 3.1, Numeric %: 0%, Length: 18

**DGA Domain Examples:**
- `xj3k2mzpqw.com` - Entropy: 3.8, Numeric %: 20%, Length: 14
- `aabc123xyz.net` - Entropy: 3.6, Numeric %: 27%, Length: 14
- `qwertyuiop.tk` - Entropy: 3.4, Numeric %: 0%, Length: 14

**Exfiltration Domain Examples:**
- `abcd1234efgh5678.example.com` - Entropy: 3.5, Numeric %: 40%, Length: 31
- `base64encodeddata.tunnel.net` - Entropy: 3.7, Numeric %: 15%, Length: 33

---

## Appendix D: Resource Requirements

**Minimum System Requirements:**
- **RAM:** 8 GB
- **Storage:** 10 GB free space
- **CPU:** Quad-core processor
- **Python:** 3.9+

**Recommended System Requirements:**
- **RAM:** 16+ GB
- **Storage:** 20 GB SSD
- **CPU:** 8+ core processor
- **GPU:** Optional (LightGBM CPU version used)
- **Python:** 3.10+

**Processing Time Estimates:**
- **Data Loading:** 2-5 minutes (full dataset)
- **Preprocessing:** 1-3 minutes
- **Feature Selection:** 2-5 minutes
- **Model Training:** 1-3 minutes
- **Evaluation:** <1 minute
- **Total Pipeline:** 8-15 minutes (first run), 1-2 minutes (cached)

---

**Document Version:** 1.0  
**Last Updated:** October 17, 2025  
**Maintained By:** DNS Spoofing Detection Project Team  
**Contact:** [Project Repository]

---

*This dataset description is provided for academic and research purposes. Please use responsibly and cite appropriately.*
