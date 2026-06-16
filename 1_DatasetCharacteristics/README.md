# Dataset Characteristics

**[ETL Notebook](exploratory_data_analysis.ipynb)**
**[EDA Notebook](EDA_updated.ipynb)**

## Dataset Information

### Data Source
1. **[RIPE Atlas Daily Measurement](https://data-store.ripe.net/datasets/atlas-daily-dumps/2026-04-29/)**
	[Explanation of Data Fields](https://atlas.ripe.net/docs/apis/measurement-result-format/version-5000#version-5000-ping-v6-ping)
	
2. [GeoIP Lite2 databases](https://www.maxmind.com/en/home)

3. [Google Drive copy of the Dataset for easier asccess](https://drive.google.com/drive/folders/1H3eyh8CgYErbDcPGOH9lEslPub40fL9V?usp=sharing)

🚧🚧🚧🚧🚧🚧🚧🚧🚧🚧🚧
  *WORK IN PROGRESS*
🚧🚧🚧🚧🚧🚧🚧🚧🚧🚧🚧

### Dataset Characteristics
- **Number of Observations:** 1116767
- **Number of Features:** 41

### Target Variable/Label
- **Label Name:** avg
- **Label Type:** [Regression]
- **Label Description:** Average round-trip time (RTT) across all replies in milliseconds. 
- **Label Values:** [For regression: range of values.]
- **Label Distribution:** [Brief description of value distribution for regression]

### Feature Description

Here is table detailing the 41 features of our final `2026-05-19-lat-est.parquet` dataset. This file is renamed as `df_original.parquet` and saved inside the dataset folder.

| Feature Name | Data Type | Description |
| --- | --- | --- |
| timestamp | datetime[μs] | Unix timestamp of when the measurement was conducted, converted to datetime. |
| src_addr | str | Public-facing IPv4 address of the RIPE Atlas probe (the source). |
| dst_addr | str | Resolved destination IPv4 address that was actually pinged. |
| sent | i64 | Total number of ping packets sent. |
| rcvd | i64 | Total number of ping replies received. |
| dup | i64 | Number of duplicate ping replies received. |
| min | f64 | Minimum round-trip time (RTT) across all replies in milliseconds. |
| max | f64 | Maximum round-trip time (RTT) across all replies in milliseconds. |
| `avg` | `f64` | *Our Target Variable*. Average round-trip time (RTT) across all replies in milliseconds. |
| src_continent | str | Geographic continent of the source IP. |
| src_country | str | Geographic country of the source IP. |
| src_longitude | f64 | Longitude coordinate of the source IP. |
| src_latitude | f64 | Latitude coordinate of the source IP. |
| src_asn | i64 | Autonomous System Number (ASN) associated with the source IP. |
| src_conn_type | str | Type of connection for the source IP (e.g., isp, hosting). |
| src_provider | str | Network provider or carrier name for the source IP. |
| src_is_mobile | bool | Flag indicating if the source IP belongs to a mobile network. |
| src_is_cloud_provider | bool | Flag indicating if the source IP belongs to a cloud provider. |
| src_is_proxy | bool | Flag indicating if the source IP is a known proxy. |
| src_is_vpn | bool | Flag indicating if the source IP is a known VPN endpoint. |
| src_is_tor | bool | Flag indicating if the source IP is part of the Tor network. |
| src_is_tor_exit | bool | Flag indicating if the source IP is a Tor exit node. |
| dst_continent | str | Geographic continent of the destination IP. |
| dst_country | str | Geographic country of the destination IP. |
| dst_longitude | f64 | Longitude coordinate of the destination IP. |
| dst_latitude | f64 | Latitude coordinate of the destination IP. |
| dst_asn | i64 | Autonomous System Number (ASN) associated with the destination IP. |
| dst_conn_type | str | Type of connection for the destination IP (e.g., isp, education). |
| dst_provider | str | Network provider or carrier name for the destination IP. |
| dst_is_mobile | bool | Flag indicating if the destination IP belongs to a mobile network. |
| dst_is_cloud_provider | bool | Flag indicating if the destination IP belongs to a cloud provider. |
| dst_is_proxy | bool | Flag indicating if the destination IP is a known proxy. |
| dst_is_vpn | bool | Flag indicating if the destination IP is a known VPN endpoint. |
| dst_is_tor | bool | Flag indicating if the destination IP is part of the Tor network. |
| dst_is_tor_exit | bool | Flag indicating if the destination IP is a Tor exit node. |
| hop_count | i64 | Number of routing hops between the source and destination (from traceroute). |
| distance_km | f64 | Calculated geographic distance between source and destination in kilometers (Haversine formula). |
| src_prefix_8 | str | The /8 network prefix (first octet) of the source IP. |
| src_prefix_16 | str | The /16 network prefix (first two octets) of the source IP. |
| dst_prefix_8 | str | The /8 network prefix (first octet) of the destination IP. |
| dst_prefix_16 | str | The /16 network prefix (first two octets) of the destination IP. |

## Exploratory Data Analysis

The exploratory data analysis is conducted in the [EDA Notebook](EDA_updated.ipynb)notebook, which includes:

- Data loading and initial inspection
- Statistical summaries and distributions
- Missing value analysis
- Feature correlation analysis
- Data visualization and insights
- Data quality assessment
