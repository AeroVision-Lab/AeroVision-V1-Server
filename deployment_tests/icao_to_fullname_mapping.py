"""
ICAO 代码到完整机型名称的映射表

用于解决测试标签（ICAO代码）与模型输出（完整名称）不匹配的问题。
"""

# 常见机型的 ICAO 代码映射
ICAO_TO_FULLNAME = {
    # Airbus A320 Family
    "A20N": "A320neo",
    "A21N": "A321neo",
    "A319": "A319",
    "A320": "A320",
    "A321": "A321",
    
    # Airbus A330 Family
    "A332": "A330-200",
    "A333": "A330-300",
    "A332F": "A330-200F",
    
    # Airbus A340 Family
    "A342": "A340-200",
    "A343": "A340-300",
    "A345": "A340-500",
    "A346": "A340-600",
    
    # Airbus A350 Family
    "A359": "A350-900",
    "A35K": "A350-1000",
    
    # Airbus A380
    "A388": "A380",
    "A380": "A380",
    
    # Boeing 737 Classic
    "B732": "737-200",
    "B733": "737-300",
    "B734": "737-400",
    "B735": "737-500",
    
    # Boeing 737 NG
    "B736": "737-600",
    "B737": "737-700",
    "B738": "737-800",
    "B739": "737-900",
    
    # Boeing 737 MAX
    "B38M": "737 MAX 8",
    
    # Boeing 747 Family
    "B741": "747-100",
    "B742": "747-200",
    "B743": "747-300",
    "B744": "747-400",
    "B748": "747-8",
    "B748F": "747-8F",
    
    # Boeing 757 Family
    "B752": "757-200",
    "B753": "757-300",
    
    # Boeing 767 Family
    "B762": "767-200",
    "B763": "767-300",
    "B764": "767-400",
    
    # Boeing 777 Family
    "B772": "777-200",
    "B773": "777-300",
    "B77L": "777-200LR",
    "B77W": "777-300ER",
    "B77F": "777F",
    
    # Boeing 787 Family
    "B788": "787-8",
    "B789": "787-9",
    "B78X": "787-10",
    
    # Boeing 717
    "B712": "Boeing_717",
    
    # Other Boeing
    "B707": "707-320",
    "B722": "727-200",
    "B727": "727-200",
    "B752": "757-200",
    
    # McDonnell Douglas
    "MD11": "MD-11",
    "MD80": "MD-80",
    "MD87": "MD-87",
    "MD90": "MD-90",
    "DC10": "DC-10",
    "DC9": "DC-9-30",
    "DC8": "DC-8",
    
    # ATR
    "AT42": "ATR-42",
    "AT72": "ATR-72",
    
    # Embraer Regional Jets
    "E170": "E-170",
    "E190": "E-190",
    "E195": "E-195",
    "ER35": "ERJ_135",
    "ER45": "ERJ_145",
    "E120": "EMB-120",
    
    # Bombardier / Canadair
    "CRJ2": "CRJ-200",
    "CRJ7": "CRJ-700",
    "CRJ9": "CRJ-900",
    
    # Cessna
    "C172": "Cessna_172",
    "C208": "Cessna_208",
    "C525": "Cessna_525",
    "C560": "Cessna_560",
    
    # Dassault
    "FA20": "Falcon_2000",
    "FA90": "Falcon_900",
    
    # Fokker
    "F100": "Fokker_100",
    "F50": "Fokker_50",
    "F70": "Fokker_70",
    
    # Hawker / Beechcraft
    "B190": "Beechcraft_1900",
    "BA25": "BAE-125",
    "BJ01": "Cessna_525",
    
    # BAE Systems
    "B462": "BAE_146-200",
    "B463": "BAE_146-300",
    
    # Saab
    "SF34": "Saab_340",
    "S2000": "Saab_2000",
    
    # Other
    "C130": "C-130",
    "DC3": "DC-3",
    "DC6": "DC-6",
    "DH82": "DH-82",
    "DHC1": "DHC-1",
    "DHC6": "DHC-6",
    "DHC8": "DHC-8-100",
    "DOR3": "Dornier_328",
    "GLF4": "Gulfstream_IV",
    "GLF5": "Gulfstream_V",
    "IL76": "Il-76",
    "L101": "L-1011",
    "ME20": "Metroliner",
    "PA28": "PA-28",
    "SR20": "SR-20",
    "TU20": "Tu-204",
    "TU21": "Tu-214",
    "TU33": "Tu-334",
    "TU51": "Tu-154",
    "TU15": "Tu-154",
    "YK42": "Yak-42",
    
    # Chinese Aircraft
    "C919": "C919",
    
    # Legacy/General Aviation
    "CH60": "Challenger_600",
    "GL64": "Global_Express",
    "MB20": "Model_B200",
    
    # Special Cases (direct matches already exist in model)
    "An-12": "An-12",
    "Tornado": "Tornado",
    "Spitfire": "Spitfire",
    "Hawk_T1": "Hawk_T1",
    "Eurofighter_Typhoon": "Eurofighter_Typhoon",
    "F-16A_B": "F-16A_B",
    "F_A-18": "F_A-18",
}


def get_fullname(icao_code: str) -> str:
    """
    将 ICAO 代码转换为完整机型名称
    
    Args:
        icao_code: ICAO 代码 (如 A332, B77W)
    
    Returns:
        完整机型名称 (如 A330-200, 777-300ER)
        如果没有找到映射，返回原始代码
    """
    return ICAO_TO_FULLNAME.get(icao_code, icao_code)


def get_icao_code(fullname: str) -> str:
    """
    将完整机型名称转换为 ICAO 代码
    
    Args:
        fullname: 完整机型名称 (如 A330-200, 777-300ER)
    
    Returns:
        ICAO 代码 (如 A332, B77W)
        如果没有找到映射，返回原始名称
    """
    # 创建反向映射
    reverse_map = {v: k for k, v in ICAO_TO_FULLNAME.items()}
    return reverse_map.get(fullname, fullname)
