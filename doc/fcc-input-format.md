

# FCC Input Format

Overview of the CSV files in input/fcc/.
All data files included here are public information available on the FCC's website.

## `Domain.csv`

- File has no header row.
- Column 1 is the literal string `DOMAIN`.
- Column 2 is the station identifier (always a number).
- Remaining columns list the channels that station is allowed to use.

Example:

DOMAIN,87,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,38,39,40,41,42,43,44,45,46,47,48,49,50,51

## `Interference_Paired.csv`

- Each row lists the incompatibilities for a single "subject" station.
- Column 1 is the constraint type:
  - `CO`: same-channel conflicts.
  - `ADJ±n`: conflicts with channels offset by `n` (e.g., `ADJ-1` is one channel below, `ADJ+2` is two channels above).
- Column 2 is the channel assigned to the subject station.
- Column 3 is the channel the listed peer stations cannot use (equal to column 2 for `CO`, offset by `n` for `ADJ±n`).
- Column 4 is the subject station ID.
- Columns 5 and higher list the peer station IDs that conflict with the subject station; the row does **not** imply the peers conflict with each other.
- Files include both directions of every constraint (they are symmetric).

Example:

CO,3,3,1500053,1500054,1500055
CO,3,3,1500054,1500053,1500055
ADJ-1,4,3,87,50194,86532
ADJ+1,3,4,87,50194,86532

## `parameters.csv`

- Includes detailed metadata for each station: facility ID, call sign, location, and numerous study parameters.
- Latitude and longitude columns are most relevant for geographic calculations.
- Column definitions and formatting details are documented in `doc/fcc-input-parameters-csv.md`.

Excerpt:

tvstudy v1.3.2
Database,"localhost"
CDBS,"10/16/2015 (2)  2015/10/20 UCM"
Study,"20151020UCM-SampleData"
Start,"2015.11.02 13:18:47"
Stations,,,,,,,,,,,,,,,,,,,,,,,,,,,,NoiseLimited,,TerrainLimited,,InterferenceFree,,,Antenna,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,ExtraFVs
FacID,SrcKey,DTSKey,Site,FileNumber,AppID,Country,D,U,Call,Ch,FromCh,City,St,Lat,Lon,DTSDist,RCAMSL,HAAT,ERP,DA,AntID,Rot,Tilt,Offset,Mask,Type,InCountry,Area,Population,Area,Population,Area,Population,,0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350,,Az,FV,Az,FV,Az,FV,Az,FV,Az,FV,Az,FV
21488,378,,0,"BLCDT20110307ACV",1565982,US,1,1,"KYES-TV",5,,"ANCHORAGE",AK,61.335735,149.515214,,614.500000,277.000000,15.0,DA,93311,0.000000,0.0,0,0,1,US,31860.2,392105,30393.4,391590,30393.4,391590,,0.987000,1.000000,0.977000,0.900000,0.791000,0.651000,0.506000,0.360000,0.232000,0.179000,0.149000,0.132000,0.134000,0.168000,0.196000,0.291000,0.432000,0.579000,0.726000,0.847000,0.943000,1.000000,0.991000,0.966000,0.898000,0.796000,0.681000,0.549000,0.430000,0.387000,0.400000,0.481000,0.619000,0.738000,0.851000,0.934000,

# post_auction_parameters.csv

This contains the new assignment of channels to the stations that stayed on the air post-auction.

The table fields are as follows:

- **FacID**: Numeric FCC Facility ID (primary key)
- **Site**: FCC tower site index
- **Call**: Station callsign (e.g., KYES-TV)
- **Ch**: Post-auction RF channel
- **PC**: Pre-auction RF channel
- **City**: City of license
- **St**: State/territory (2-letter code)
- **Lat / Lon**: Transmitter coordinates (FCC integer format)
- **RCAMSL**: Radiation Center Above Mean Sea Level (m)
- **HAAT**: Height Above Average Terrain (m)
- **ERP**: Effective Radiated Power (kW)
- **DA**: Directional antenna (yes/no)
- **AntID**: Antenna ID number
- **Az**: Antenna azimuth (deg)
