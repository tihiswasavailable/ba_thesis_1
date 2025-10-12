# config/settings.py
"""
Main configuration file for the cookie scraper pipeline
"""

import os
from pathlib import Path

class Config:
    # Debug settings
    DEBUG_MODE = True
    VERBOSE_LOGGING = True
    TAKE_SCREENSHOTS = True
    
    # Browser settings
    PAGE_LOAD_TIMEOUT = 30
    INITIAL_WAIT = 5  # Wait after page load
    INTERACTION_WAIT = 2  # Wait after user simulation
    MAX_RETRIES = 2
    
    # Parallel processing
    MAX_WORKERS = 2  # Reduce for stability
    PROCESS_TIMEOUT = 180  # 3 minutes per URL
    
    # Data quality
    MIN_BUTTON_TEXT_LENGTH = 2
    MAX_BUTTON_TEXT_LENGTH = 150
    MIN_BANNER_HEIGHT = 30
    MIN_BANNER_WIDTH = 200
    
    # File paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    MODELS_DIR = DATA_DIR / "models"
    LOGS_DIR = BASE_DIR / "logs"
    SCREENSHOTS_DIR = BASE_DIR / "screenshots"
    
    # Create directories
    for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR, SCREENSHOTS_DIR]:
        dir_path.mkdir(exist_ok=True)
    
    # Output filenames
    @staticmethod
    def get_output_filename(base_name, extension="csv"):
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{base_name}_{timestamp}.{extension}"

# config/selectors.py
"""
CMP and CSS selectors for banner detection
"""

# OneTrust
ONETRUST_SELECTORS = [
    '#onetrust-banner-sdk', '#onetrust-consent-sdk', '.onetrust-pc-sdk',
    '#onetrust-policy-text', '.onetrust-policy-text', '#onetrust-pc-sdk'
]

# Cookiebot
COOKIEBOT_SELECTORS = [
    '#CybotCookiebotDialog', '.CybotCookiebotDialog', 
    '#CybotCookiebotDialogBodyUnderlay', '#CybotCookiebotDialogBodyContentText',
    '.CybotCookiebotDialogBodyButton'
]

# Usercentrics (German)
USERCENTRICS_SELECTORS = [
    '#usercentrics-cmp', '.usercentrics-cmp', '#usercentrics-root',
    '[data-testid="uc-banner"]', '[data-testid="uc-container"]',
    '.usercentrics-banner', '#uc-banner', '.uc-banner'
]

# Consentmanager (German)
CONSENTMANAGER_SELECTORS = [
    '#cmpbox', '.cmpbox', '#cmpwrapper', '.cmpwrapper',
    '[id^="cmp"]', '[class*="consentmanager"]'
]

# Didomi
DIDOMI_SELECTORS = [
    '#didomi-host', '.didomi-consent-popup', '.didomi-popup',
    '.didomi-banner', '.didomi-notice', '#didomi-notice',
    '#didomi-popup', '.didomi-screen-medium'
]

# Sourcepoint (Zeit.de, Spiegel, etc.)
SOURCEPOINT_SELECTORS = [
    '[id^="sp_message_container_"]', '.sp-message', '.sp-message-open',
    '.sp-message-container', '#sp-cc', '.sp_choice_type_11',
    '[id^="sp_message_"]', '.sp_message_container'
]

# Quantcast
QUANTCAST_SELECTORS = [
    '#qcCmpUi', '.qc-cmp-ui', '.qc-cmp2-container',
    '.qc-cmp-ui-container', '#qc-cmp2-container'
]

# Generic German CMPs
GERMAN_CMP_SELECTORS = [
    '.cookie-layer', '.cookie-overlay', '.datenschutz-layer',
    '#cookie-layer', '#datenschutz-banner', '.privacy-layer',
    '.dsgvo-banner', '#dsgvo-banner', '.gdpr-layer'
]

# All CMP-specific selectors
CMP_SELECTORS = (
    ONETRUST_SELECTORS + COOKIEBOT_SELECTORS + USERCENTRICS_SELECTORS +
    CONSENTMANAGER_SELECTORS + DIDOMI_SELECTORS + SOURCEPOINT_SELECTORS +
    QUANTCAST_SELECTORS + GERMAN_CMP_SELECTORS
)

# Generic selectors
GENERIC_SELECTORS = [
    # Class-based
    '[class*="consent-manager"]', '[class*="privacy-manager"]',
    '[class*="cookie-layer"]', '[class*="cookie-overlay"]',
    '[class*="gdpr-layer"]', '[class*="privacy-layer"]',
    '[class*="cookie-hint"]', '[class*="cookie-info"]',
    '[class*="consent-layer"]', '[class*="privacy-notice"]',
    
    # Data attributes
    '[data-cookie-banner]', '[data-consent-banner]',
    '[data-privacy-banner]', '[data-gdpr-banner]',
    '[data-consent]', '[data-cookie]', '[data-privacy]',
    '[data-testid*="cookie"]', '[data-testid*="consent"]',
    '[data-testid*="privacy"]', '[data-testid*="banner"]',
    
    # Role-based
    '[role="dialog"][class*="cookie"]', '[role="dialog"][class*="consent"]',
    '[role="dialog"][class*="privacy"]', '[role="alertdialog"]',
    '[role="banner"][class*="cookie"]', '[role="banner"][class*="consent"]',
    
    # Aria labels
    '[aria-label*="cookie" i]', '[aria-label*="consent" i]',
    '[aria-label*="privacy" i]', '[aria-label*="datenschutz" i]',
    
    # German terms
    '[class*="datenschutz"]', '[class*="einwilligung"]',
    '[class*="zustimmung"]', '[class*="cookies"]'
]

# iFrame selectors
IFRAME_SELECTORS = [
    'iframe[title*="cookie" i]', 'iframe[title*="privacy" i]',
    'iframe[src*="cookie" i]', 'iframe[src*="consent" i]',
    'iframe[id*="cookie" i]', 'iframe[class*="cookie" i]',
    'iframe[name*="cmp" i]', 'iframe[id*="cmp" i]',
    'iframe[src*="tcfApiLocator"]', 'iframe[name*="tcfApiLocator"]',
    'iframe[src*="didomi"]', 'iframe[src*="onetrust"]',
    'iframe[src*="cookiebot"]', 'iframe[src*="sourcepoint"]'
]

# config/keywords.py
"""
Keywords for privacy classification and validation
"""

# Cookie-related keywords for validation
COOKIE_KEYWORDS = [
    'cookie', 'consent', 'einwilligung', 'zustimm', 'akzeptier', 'ablehnen',
    'verweiger', 'erlauben', 'zulassen', 'einstellung', 'präferenz', 'datenschutz',
    'privacy', 'accept', 'decline', 'reject', 'allow', 'deny', 'agree',
    'disagree', 'opt-in', 'opt-out', 'manage', 'customize', 'settings',
    'notwendig', 'erforderlich', 'essenziell', 'funktional', 'marketing',
    'analytics', 'tracking', 'personalisier', 'werbung', 'statistik',
    'essential', 'necessary', 'required', 'functional', 'advertising',
    'alle', 'all', 'nur', 'only', 'basic', 'advanced', 'erweitert',
    'speichern', 'save', 'bestätigen', 'confirm', 'fortfahren', 'continue',
    'verstanden', 'geht klar', 'in ordnung', 'einverstanden', 'ok',
    'werbefrei', 'werbefinanziert', 'auswählen', 'wählen', 'bestätigung'
]

# Exclusion keywords
EXCLUDE_KEYWORDS = [
    'menü', 'menu', 'navigation', 'nav', 'login', 'anmelden', 'konto', 'account',
    'suche', 'search', 'warenkorb', 'cart', 'sprache', 'language', 'währung',
    'home', 'startseite', 'impressum', 'kontakt', 'about', 'über', 'hilfe', 'help',
    'newsletter', 'news', 'artikel', 'article', 'produkt', 'product', 'kategorie',
    'triebwerk', 'boeing', 'zwischenfall', 'nachrichten', 'sport', 'fußball'
]

# Privacy classification keywords
PRIVACY_FRIENDLY_KEYWORDS = [
    "nur notwendige", "nur erforderliche", "nur essentielle", "ablehnen",
    "alle ablehnen", "einwilligung ablehnen", "cookies ablehnen", "verweigern",
    "nein danke", "nicht zustimmen", "zurückweisen", "minimal",
    "notwendige cookies", "erforderliche cookies", "funktionale cookies",
    "technisch notwendig", "technisch erforderlich", "grundlegend",
    "decline", "decline all", "reject", "reject all", "deny", "deny all",
    "refuse", "refuse all", "no thanks", "not consent", "opt out",
    "essential only", "necessary only", "required only", "functional only",
    "strictly necessary", "technically required", "basic cookies",
    "einwilligung ablehnen", "werbefrei"
]

PRIVACY_RISKY_KEYWORDS = [
    "alle akzeptieren", "alles akzeptieren", "akzeptieren", "zustimmen",
    "einverstanden", "ok", "geht klar", "verstanden", "in ordnung",
    "cookies akzeptieren", "alle cookies", "ja", "ja akzeptieren",
    "einwilligung erteilen", "alle erlauben", "erlauben", "alle zulassen",
    "accept all", "accept", "agree", "allow", "allow all", "consent",
    "yes", "okay", "got it", "understand", "fine", "sure",
    "enable all", "all cookies", "i agree", "i consent",
    "werbefinanziert", "alle auswählen"
]

NEUTRAL_KEYWORDS = [
    "einstellungen", "konfigurieren", "anpassen", "präferenzen",
    "mehr informationen", "weitere informationen", "details", "optionen",
    "verwalten", "auswählen", "individuell", "benutzerdefiniert",
    "cookie-einstellungen", "datenschutz-einstellungen", "erweitert",
    "settings", "preferences", "configure", "manage", "customize",
    "learn more", "more info", "details", "options", "advanced",
    "cookie settings", "privacy settings", "choose", "select",
    "speichern", "bestätigen", "weiter", "fortfahren", "continue"
]