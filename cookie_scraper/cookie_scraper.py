#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCRIPT 1: Cookie Banner Scraper - Sammelt Banner mit allen Buttons
Fokus: Banner-Level Daten sammeln (nicht nur einzelne Buttons)
"""

import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import time
import logging
from urllib.parse import urlparse
from datetime import datetime
import json
import concurrent.futures

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# URLs aus der bereitgestellten Liste
from dach_urls_list import ALLE_NEUEN_URLS

class CookieBannerScraper:
    """
    Sammelt komplette Cookie-Banner (nicht nur einzelne Buttons)
    Jedes Banner enthält ALLE Buttons für späteren Vergleich
    """
    
    def __init__(self):
        self.stats = {'processed': 0, 'successful': 0, 'banners': 0}
        self.failed_urls = []
        self.debug_mode = True
        
        # VOLLSTÄNDIGE Selektor-Liste (wie dein bewährtes Original)
        self.banner_selectors = [
            # CMP-spezifische Selektoren
            '#onetrust-banner-sdk', '#onetrust-consent-sdk', '.onetrust-pc-sdk',
            '#onetrust-policy-text', '.onetrust-policy-text',
            
            # Cookiebot
            '#CybotCookiebotDialog', '.CybotCookiebotDialog', '#CybotCookiebotDialogBodyUnderlay',
            '#CybotCookiebotDialogBodyContentText', '.CybotCookiebotDialogBodyButton',
            
            # Usercentrics (häufig in Deutschland)
            '#usercentrics-cmp', '.usercentrics-cmp', '#usercentrics-root',
            '[data-testid="uc-banner"]', '[data-testid="uc-container"]',
            '.usercentrics-banner', '#uc-banner', '.uc-banner',
            
            # Consentmanager (deutsch)
            '#cmpbox', '.cmpbox', '#cmpwrapper', '.cmpwrapper',
            '[id^="cmp"]', '[class*="consentmanager"]',
            
            # Didomi (weit verbreitet in DE)
            '#didomi-host', '.didomi-consent-popup', '.didomi-popup',
            '.didomi-banner', '.didomi-notice', '#didomi-notice',
            '#didomi-popup', '.didomi-screen-medium',
            
            # Quantcast
            '#qcCmpUi', '.qc-cmp-ui', '.qc-cmp2-container',
            '.qc-cmp-ui-container', '#qc-cmp2-container',
            
            # Sourcepoint (Spiegel, Bild, etc.)
            '[id^="sp_message_container_"]', '.sp-message', '.sp-message-open',
            '.sp-message-container', '#sp-cc', '.sp_choice_type_11',
            '[id^="sp_message_"]', '.sp_message_container',
            
            # TrustArc
            '#truste-consent-track', '.truste-consent-track', '#consent-pref-link',
            '#trustarc-banner', '.trustarc-banner',
            
            # Borlabs Cookie (WordPress Plugin)
            '.borlabs-cookie', '#BorlabsCookieBox', '#BorlabsCookie',
            '.borlabs-cookie-banner', '#borlabs-cookie-banner',
            
            # Cookie Consent (beliebte Bibliothek)
            '.cc-window', '.cc-banner', '.cookie-consent', '.cookieconsent',
            '#cookieconsent', '.cc-floating', '.cc-bottom', '.cc-top',
            
            # Klaro
            '.klaro', '.cookie-notice', '.cookie-modal', '#klaro',
            '.klaro-modal', '#klaro-manager',
            
            # Deutsche/österreichische CMPs
            '.cookie-layer', '.cookie-overlay', '.datenschutz-layer',
            '#cookie-layer', '#datenschutz-banner', '.privacy-layer',
            '.dsgvo-banner', '#dsgvo-banner', '.gdpr-layer',
            
            # NEUE Selektoren basierend auf Screenshots
            '.message-container', '.cmp-banner', '.message-overlay',
            '.type-modal', '.message-safe-area-holder', '.message-safe-area',
            '[class*="message-container"]', '[class*="cmp-banner"]',
            '[class*="message-overlay"]', '[class*="type-modal"]',
            '[class*="message-safe-area"]', '[class*="message-component"]',
            
            # Heise-spezifische Selektoren
            '[class*="consent"]', '[id*="consent"]',
            '.modal[class*="consent"]', '.overlay[class*="consent"]',
            
            # SEHR WICHTIG: Generische Pattern
            '[class*="consent-manager"]', '[class*="privacy-manager"]',
            '[class*="cookie-layer"]', '[class*="cookie-overlay"]',
            '[class*="gdpr-layer"]', '[id*="gdpr"]', '[class*="gdpr"]',
            '[class*="privacy-layer"]', '[class*="cookie-hint"]',
            '[class*="cookie-info"]', '[class*="consent-layer"]',
            '[class*="privacy-notice"]', '[id*="privacy-banner"]',
            
            # Pattern basierend auf Screenshots
            '[class*="message"]', '[class*="notice"]', '[class*="banner"]',
            '[class*="modal"]', '[class*="overlay"]', '[class*="dialog"]',
            '[class*="popup"]', '[class*="layer"]', '[class*="container"]',
            '[class*="component"]', '[class*="widget"]', '[class*="panel"]',
            '[class*="box"]', '[class*="frame"]', '[class*="window"]',
            
            # Data-Attribute (sehr wichtig für moderne CMPs)
            '[data-cookie-banner]', '[data-consent-banner]',
            '[data-privacy-banner]', '[data-gdpr-banner]',
            '[data-consent]', '[data-cookie]', '[data-privacy]',
            '[data-testid*="cookie"]', '[data-testid*="consent"]',
            '[data-testid*="privacy"]', '[data-testid*="banner"]',
            '[data-cy*="cookie"]', '[data-cy*="consent"]',
            
            # Role-basierte Selektoren
            '[role="dialog"][class*="cookie"]', '[role="dialog"][class*="consent"]',
            '[role="dialog"][class*="privacy"]', '[role="alertdialog"]',
            '[role="banner"][class*="cookie"]', '[role="banner"][class*="consent"]',
            
            # Aria-Labels
            '[aria-label*="cookie" i]', '[aria-label*="consent" i]',
            '[aria-label*="privacy" i]', '[aria-label*="datenschutz" i]',
            '[aria-describedby*="cookie"]', '[aria-describedby*="consent"]',
            
            # Deutsche Begriffe
            '[class*="datenschutz"]', '[class*="einwilligung"]',
            '[class*="zustimmung"]', '[class*="cookies"]',
            '[id*="datenschutz"]', '[id*="einwilligung"]',
            
            # Modal/Overlay Pattern
            '.modal[class*="cookie"]', '.modal[class*="privacy"]',
            '.popup[class*="cookie"]', '.overlay[class*="consent"]',
            '.dialog[class*="cookie"]', '.lightbox[class*="consent"]',
            
            # Positionierung
            '.fixed[class*="cookie"]', '.sticky[class*="privacy"]',
            '[style*="position: fixed"][class*="cookie"]',
            '[style*="position: absolute"][class*="consent"]',
            '[style*="z-index"][class*="cookie"]',
            
            # Framework-spezifische
            '.v-dialog[class*="cookie"]', '.el-dialog[class*="consent"]',
            '.ant-modal[class*="cookie"]', '.mui-dialog[class*="privacy"]',
            
            # Weitere generische Pattern
            'section[class*="consent"]', 'section[class*="privacy"]',
            'aside[class*="consent"]', 'aside[class*="privacy"]',
            'footer[class*="cookie"]', 'header[class*="consent"]'
        ]

    def setup_driver(self):
        """Chrome Driver Setup"""
        chrome_options = Options()
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        
        prefs = {"profile.default_content_setting_values.notifications": 2}
        chrome_options.add_experimental_option("prefs", prefs)
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

        driver = webdriver.Chrome(options=chrome_options)
        driver.set_page_load_timeout(30)
        driver.implicitly_wait(0)
        
        return driver

    def trigger_banners(self, driver):
        """Trigger Cookie-Banner (vereinfacht)"""
        try:
            # Standard-Trigger
            driver.execute_script("window.scrollTo(0, 100);")
            time.sleep(1)
            driver.execute_script("window.scrollTo(0, 0);")
            
            # Event-Trigger
            driver.execute_script("""
                window.dispatchEvent(new Event('scroll'));
                window.dispatchEvent(new Event('resize'));
                document.dispatchEvent(new Event('DOMContentLoaded'));
            """)
            
            time.sleep(3)
        except Exception as e:
            logger.debug(f"Error triggering banners: {e}")

    def find_cookie_banners(self, driver):
        """Finde alle Cookie-Banner auf der Seite"""
        banners = []
        
        # Hauptsuche
        try:
            selector = ', '.join(self.banner_selectors)
            elements = driver.find_elements(By.CSS_SELECTOR, selector)
            
            for element in elements:
                if self.is_valid_banner(element):
                    banners.append(element)
        except Exception as e:
            logger.debug(f"Banner search failed: {e}")
        
        # JavaScript-Fallback
        if not banners:
            try:
                js_banners = driver.execute_script("""
                    var banners = [];
                    var allElements = document.querySelectorAll('*');
                    
                    for (var i = 0; i < allElements.length; i++) {
                        var el = allElements[i];
                        var text = (el.textContent || '').toLowerCase();
                        var className = (el.className || '').toLowerCase();
                        
                        if ((text.includes('cookie') || text.includes('datenschutz') || 
                             className.includes('cookie') || className.includes('consent')) &&
                            el.querySelectorAll('button, [role="button"], a, div[onclick]').length >= 2) {
                            banners.push(el);
                        }
                    }
                    return banners;
                """)
                
                for element in js_banners:
                    if self.is_valid_banner(element):
                        banners.append(element)
                        
            except Exception as e:
                logger.debug(f"JavaScript fallback failed: {e}")
        
        return banners

    def is_valid_banner(self, element):
        """AGGRESSIVE Banner-Validierung (wie dein Original)"""
        try:
            if not element:
                return False

            # LOCKERE Größenprüfung (wie dein Original)
            try:
                size = element.size
                if size['height'] < 5 or size['width'] < 50:
                    if self.debug_mode:
                        logger.debug(f"Banner too small: {size}")
                    return False
            except:
                pass

            # Text-Content sammeln (erweitert)
            text_content = ""
            try:
                text_content = element.text.lower().strip()
            except:
                pass
                
            # Auch Attribute prüfen
            aria_label = element.get_attribute('aria-label') or ''
            title = element.get_attribute('title') or ''
            class_name = element.get_attribute('class') or ''
            id_attr = element.get_attribute('id') or ''
            
            combined_text = f"{text_content} {aria_label} {title} {class_name} {id_attr}".lower()
            
            # ERWEITERTE Banner-Keywords (wie dein Original)
            banner_keywords = [
                'cookie', 'consent', 'privacy', 'datenschutz', 'gdpr', 'dsgvo', 
                'tracking', 'privatsphäre', 'einwilligung', 'akzeptieren', 'ablehnen',
                'zustimmung', 'erlauben', 'verweigern', 'speichern wir', 'verwenden wir',
                'analyse', 'marketing', 'werbung', 'personalisiert', 'funktional', 
                'statistik', 'essential', 'necessary', 'datenverarbeitung',
                'cookies und', 'diese website', 'ihre daten', 'werbefinanziert', 
                'werbefrei', 'partner', 'message', 'notice', 'banner', 'modal', 
                'overlay', 'dialog', 'popup', 'layer', 'container', 'component',
                'sp-message', 'onetrust', 'cookiebot', 'usercentrics', 'didomi'
            ]

            if not any(keyword in combined_text for keyword in banner_keywords):
                if self.debug_mode:
                    logger.debug(f"No banner keywords found in: {combined_text[:150]}")
                return False

            # SEHR ERWEITERTE Button-Suche (wie dein Original)
            button_selectors = [
                'button', 'a[role="button"]', 'input[type="submit"]', 'input[type="button"]', 
                'div[role="button"]', 'span[role="button"]', 'a[onclick]', 'div[onclick]',
                'a[href*="consent"]', 'a[href*="cookie"]', '[data-action]', 
                '.btn', '.button', '[class*="btn"]', '[class*="button"]',
                '.message-button', '.sp_choice_type_11', '.message-component',
                '[class*="choice"]', '[class*="message-button"]',
                '[class*="message"]', '[class*="notice"]', '[class*="banner"]',
                '[class*="modal"]', '[class*="overlay"]', '[class*="dialog"]',
                '[class*="popup"]', '[class*="layer"]', '[class*="container"]',
                '[class*="component"]', '[class*="widget"]', '[class*="panel"]',
                'div[class*="accept"]', 'div[class*="reject"]', 'div[class*="decline"]',
                'div[class*="consent"]', 'div[class*="cookie"]', 'div[class*="agree"]',
                'span[class*="accept"]', 'span[class*="reject"]', 'span[class*="consent"]',
                '[style*="cursor: pointer"]', '[tabindex]:not([tabindex="-1"])',
                '[role="button"]', '[aria-label*="accept"]', '[aria-label*="reject"]'
            ]
            
            buttons = element.find_elements(By.CSS_SELECTOR, ', '.join(button_selectors))
            
            # FALLBACK: Aggressive Suche nach clickable Elements (wie dein Original)
            if not buttons:
                potential_buttons = element.find_elements(By.CSS_SELECTOR, 'div, span, a, p')
                for pot_btn in potential_buttons:
                    try:
                        btn_text = pot_btn.text.strip().lower()
                        if (btn_text and len(btn_text) > 2 and len(btn_text) < 80 and
                            any(keyword in btn_text for keyword in 
                                ['akzep', 'ableh', 'zustimm', 'cookie', 'einstellung', 'accept', 
                                'reject', 'consent', 'ok', 'ja', 'nein', 'alle', 'nur', 'settings',
                                'manage', 'allow', 'deny', 'agree', 'weiter', 'continue'])):
                            onclick = pot_btn.get_attribute('onclick')
                            cursor_style = pot_btn.value_of_css_property('cursor')
                            tabindex = pot_btn.get_attribute('tabindex')
                            
                            if (onclick or cursor_style == 'pointer' or 
                                (tabindex and tabindex != '-1') or
                                pot_btn.tag_name.lower() in ['a', 'button']):
                                buttons.append(pot_btn)
                                if len(buttons) >= 5:
                                    break
                    except:
                        continue

            if not buttons:
                if self.debug_mode:
                    logger.debug("No buttons found in potential banner")
                return False
                
            # LOCKERE Button-Validation (mindestens 1 gültiger Button statt 2)
            valid_buttons = 0
            for btn in buttons:
                try:
                    btn_text = btn.text.strip()
                    if not btn_text:
                        btn_text = (btn.get_attribute('aria-label') or 
                                btn.get_attribute('value') or 
                                btn.get_attribute('title') or 
                                btn.get_attribute('alt') or '').strip()
                        
                    if self.is_valid_button_text(btn_text):
                        valid_buttons += 1
                        if self.debug_mode:
                            logger.debug(f"Valid cookie button found: '{btn_text}'")
                except:
                    continue
                    
            # REDUZIERT: Mindestens 1 Button (statt 2)
            if valid_buttons == 0:
                if self.debug_mode:
                    logger.debug("No valid cookie buttons found in banner")
                return False
                
            if self.debug_mode:
                logger.info(f"✅ Valid banner found with {valid_buttons} cookie buttons")
            return True

        except Exception as e:
            if self.debug_mode:
                logger.debug(f"Error validating banner element: {e}")
            return False

    def extract_banner_data(self, banner_element, url):
        """ERWEITERTE Banner-Extraktion mit ALLEN Buttons (wie dein Original)"""
        domain = urlparse(url).netloc
        
        # SEHR ERWEITERTE Button-Selektoren (wie dein Original)
        button_selectors = [
            'button', 'a[role="button"]', 'input[type="submit"]', 'input[type="button"]',
            'div[role="button"]', 'span[role="button"]', '[onclick]', '[data-action]', 
            '.btn', '.button', '[class*="btn"]', '[class*="button"]', 
            '[tabindex]:not([tabindex="-1"])', 'a[href*="consent"]', 'a[href*="cookie"]',
            
            # CMP-spezifische Button-Selektoren
            '.message-button', '.sp_choice_type_11', '.message-component',
            '[class*="choice"]', '[class*="message-button"]', '[class*="consent-button"]',
            
            # Erweiterte Button-Pattern
            '[class*="message"]', '[class*="notice"]', '[class*="banner"]',
            '[class*="modal"]', '[class*="overlay"]', '[class*="dialog"]',
            '[class*="popup"]', '[class*="layer"]', '[class*="container"]',
            '[class*="component"]', '[class*="widget"]', '[class*="panel"]',
            '[class*="box"]', '[class*="frame"]', '[class*="window"]',
            '[class*="safe-area"]', '[class*="type-modal"]',
            
            # DIV/SPAN-basierte Cookie-Buttons
            'div[class*="accept"]', 'div[class*="reject"]', 'div[class*="decline"]',
            'div[class*="consent"]', 'div[class*="cookie"]', 'div[class*="agree"]',
            'span[class*="accept"]', 'span[class*="reject"]', 'span[class*="consent"]',
            
            # Clickable Elemente
            '[style*="cursor: pointer"]', '[style*="cursor:pointer"]',
            '[role="button"]', '[aria-label*="accept"]', '[aria-label*="reject"]',
            '[aria-label*="cookie"]', '[aria-label*="consent"]'
        ]
        
        buttons = banner_element.find_elements(By.CSS_SELECTOR, ', '.join(button_selectors))
        
        # FALLBACK: Erweiterte Suche (wie dein Original)
        if not buttons:
            potential_buttons = banner_element.find_elements(By.CSS_SELECTOR, 'div, span, a, p')
            for pot_btn in potential_buttons:
                try:
                    btn_text = pot_btn.text.strip().lower()
                    if (btn_text and len(btn_text) > 2 and len(btn_text) < 80 and
                        any(keyword in btn_text for keyword in 
                            ['akzep', 'ableh', 'zustimm', 'cookie', 'einstellung', 'accept', 
                            'reject', 'consent', 'ok', 'ja', 'nein', 'alle', 'nur', 'settings',
                            'manage', 'allow', 'deny', 'agree', 'weiter', 'continue'])):
                        onclick = pot_btn.get_attribute('onclick')
                        cursor_style = pot_btn.value_of_css_property('cursor')
                        tabindex = pot_btn.get_attribute('tabindex')
                        
                        if (onclick or cursor_style == 'pointer' or 
                            (tabindex and tabindex != '-1') or
                            pot_btn.tag_name.lower() in ['a', 'button']):
                            buttons.append(pot_btn)
                            if len(buttons) >= 5:
                                break
                except:
                    continue
        
        if self.debug_mode:
            logger.debug(f"Found {len(buttons)} potential buttons in banner")
        
        # Extrahiere Button-Texte mit mehreren Methoden (wie dein Original)
        button_texts = []
        for btn in buttons:
            try:
                # Mehrere Methoden zum Text-Extrahieren
                btn_text = btn.text.strip()
                if not btn_text:
                    btn_text = (btn.get_attribute('aria-label') or 
                               btn.get_attribute('value') or 
                               btn.get_attribute('title') or 
                               btn.get_attribute('alt') or 
                               btn.get_attribute('innerText') or
                               btn.get_attribute('textContent') or '').strip()

                if btn_text and self.is_valid_button_text(btn_text):
                    button_texts.append(btn_text)
                    if self.debug_mode:
                        logger.debug(f"✅ Valid cookie button: '{btn_text}'")
                else:
                    if self.debug_mode and btn_text:
                        logger.debug(f"❌ Invalid button text: '{btn_text}'")
            except Exception as e:
                if self.debug_mode:
                    logger.debug(f"Error extracting button text: {e}")
                continue
        
        # LOCKERE Anforderung: Mindestens 1 Button (statt 2)
        if len(button_texts) < 1:
            return None
        
        # Banner-Kontext
        banner_text = ''
        try:
            banner_text = banner_element.text[:300] if banner_element.text else ''
        except:
            pass
        
        return {
            'url': url,
            'domain': domain,
            'banner_text': banner_text,
            'buttons': button_texts,  # ALLE Buttons des Banners
            'button_count': len(button_texts),
            'timestamp': datetime.now().isoformat(),
            'cmp_type': self.detect_cmp_type(banner_element)
        }

    def is_valid_button_text(self, text):
        """ERWEITERTE Button-Text-Validation (wie dein Original)"""
        if not text or not isinstance(text, str):
            return False
            
        text = text.strip()
        
        # LOCKERE Länge-Prüfung
        if len(text) < 1 or len(text) > 150:
            return False
        
        text_clean = text.replace('&amp;', '&').replace('&quot;', '"').replace('&#246;', 'ö').replace('&#160;', ' ')
        text_lower = text_clean.lower()
        
        # REDUZIERTE Ausschlusskriterien
        exclusion_patterns = [
            r'^\d+'
            ]

    def detect_cmp_type(self, element):
        """Erkenne CMP-Anbieter"""
        try:
            class_name = element.get_attribute('class') or ''
            id_attr = element.get_attribute('id') or ''
            combined = f"{class_name} {id_attr}".lower()
            
            if 'onetrust' in combined:
                return 'OneTrust'
            elif 'cookiebot' in combined or 'cybot' in combined:
                return 'Cookiebot'
            elif 'usercentrics' in combined:
                return 'Usercentrics'
            elif 'didomi' in combined:
                return 'Didomi'
            elif 'sp-message' in combined or 'sourcepoint' in combined:
                return 'Sourcepoint'
            elif 'cmp' in combined:
                return 'Consentmanager'
            elif 'borlabs' in combined:
                return 'Borlabs'
            else:
                return 'Unknown'
        except:
            return 'Unknown'

    def scrape_url(self, url):
        """Scrape einzelne URL"""
        driver = None
        start_time = time.time()
        
        try:
            if self.debug_mode:
                logger.info(f"🌐 Processing: {url}")
            
            driver = self.setup_driver()
            driver.get(url)
            
            # Warte auf Seite
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            time.sleep(5)  # Banner-Ladezeit
            
            # Suche Banner
            banners = self.find_cookie_banners(driver)
            
            # Wenn keine: Trigger
            if not banners:
                self.trigger_banners(driver)
                banners = self.find_cookie_banners(driver)
            
            # Extrahiere Banner-Daten
            banner_data_list = []
            for banner_element in banners:
                banner_data = self.extract_banner_data(banner_element, url)
                if banner_data:
                    banner_data_list.append(banner_data)
            
            elapsed = time.time() - start_time
            
            if banner_data_list:
                if self.debug_mode:
                    total_buttons = sum(len(banner['buttons']) for banner in banner_data_list)
                    logger.info(f"✅ SUCCESS: {len(banner_data_list)} banners, {total_buttons} buttons ({elapsed:.1f}s)")
                return (url, True, banner_data_list)
            else:
                if self.debug_mode:
                    logger.info(f"ℹ️ NO BANNERS: {url} ({elapsed:.1f}s)")
                return (url, True, [])
                
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"❌ ERROR: {url} - {str(e)[:100]} ({elapsed:.1f}s)")
            return (url, False, [])
        finally:
            if driver:
                try:
                    driver.quit()
                except:
                    pass

    def collect_all_banners(self, max_urls=None):
        """Sammle alle Banner von URLs"""
        logger.info("🚀 Cookie Banner Collection starting...")
        
        urls_to_process = ALLE_NEUEN_URLS[:max_urls] if max_urls else ALLE_NEUEN_URLS
        all_banner_data = []
        
        # Sequenziell (einfacher)
        for i, url in enumerate(urls_to_process):
            url_result, success, banner_data = self.scrape_url(url)
            
            if success:
                all_banner_data.extend(banner_data)
                self.stats['successful'] += 1
                self.stats['banners'] += len(banner_data)
            else:
                self.failed_urls.append(url)
            
            self.stats['processed'] += 1
            
            # Progress
            progress = ((i + 1) / len(urls_to_process)) * 100
            total_buttons = sum(len(banner['buttons']) for banner in all_banner_data)
            logger.info(f"📈 Progress: {i + 1}/{len(urls_to_process)} ({progress:.1f}%) - "
                       f"Banners: {len(all_banner_data)}, Buttons: {total_buttons}")
        
        return all_banner_data

    def save_raw_data(self, banner_data_list):
        """Speichere Roh-Daten für weitere Verarbeitung"""
        if not banner_data_list:
            logger.warning("⚠️ No banner data to save.")
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Speichere als JSON (für nächstes Script)
        json_file = f"raw_banner_data_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(banner_data_list, f, indent=2, ensure_ascii=False, default=str)
        
        # Statistiken
        total_buttons = sum(len(banner['buttons']) for banner in banner_data_list)
        domains = set(banner['domain'] for banner in banner_data_list)
        
        logger.info(f"💾 Raw data saved: {json_file}")
        logger.info(f"📊 Statistics:")
        logger.info(f"   Banners collected: {len(banner_data_list)}")
        logger.info(f"   Total buttons: {total_buttons}")
        logger.info(f"   Unique domains: {len(domains)}")
        logger.info(f"   Success rate: {self.stats['successful']}/{self.stats['processed']} URLs")
        
        return json_file

def main():
    """Hauptfunktion - Sammle Banner-Daten"""
    print("🚀 SCRIPT 1: Cookie Banner Scraper")
    print("🎯 Sammelt komplette Banner mit allen Buttons")
    print("-" * 50)
    
    scraper = CookieBannerScraper()
    
    try:
        # Sammle Banner (optional: max_urls=50 für Test)
        banner_data = scraper.collect_all_banners(max_urls=100)  # Test mit 100 URLs
        
        if banner_data:
            json_file = scraper.save_raw_data(banner_data)
            print(f"\n🎉 COLLECTION COMPLETED!")
            print(f"📁 Raw data saved to: {json_file}")
            print(f"▶️ Next: Run Script 2 to label the data")
        else:
            print("❌ No banner data collected.")
            
    except KeyboardInterrupt:
        print("\n⏹️ Collection stopped by user.")
    except Exception as e:
        print(f"💥 Error: {e}")

if __name__ == "__main__":
    main(),  # Reine Zahlen
            r'https?://', r'www\.',
            r'^(close|schließen|x|×)

    def detect_cmp_type(self, element):
        """Erkenne CMP-Anbieter"""
        try:
            class_name = element.get_attribute('class') or ''
            id_attr = element.get_attribute('id') or ''
            combined = f"{class_name} {id_attr}".lower()
            
            if 'onetrust' in combined:
                return 'OneTrust'
            elif 'cookiebot' in combined or 'cybot' in combined:
                return 'Cookiebot'
            elif 'usercentrics' in combined:
                return 'Usercentrics'
            elif 'didomi' in combined:
                return 'Didomi'
            elif 'sp-message' in combined or 'sourcepoint' in combined:
                return 'Sourcepoint'
            elif 'cmp' in combined:
                return 'Consentmanager'
            elif 'borlabs' in combined:
                return 'Borlabs'
            else:
                return 'Unknown'
        except:
            return 'Unknown'

    def scrape_url(self, url):
        """Scrape einzelne URL"""
        driver = None
        start_time = time.time()
        
        try:
            if self.debug_mode:
                logger.info(f"🌐 Processing: {url}")
            
            driver = self.setup_driver()
            driver.get(url)
            
            # Warte auf Seite
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            time.sleep(5)  # Banner-Ladezeit
            
            # Suche Banner
            banners = self.find_cookie_banners(driver)
            
            # Wenn keine: Trigger
            if not banners:
                self.trigger_banners(driver)
                banners = self.find_cookie_banners(driver)
            
            # Extrahiere Banner-Daten
            banner_data_list = []
            for banner_element in banners:
                banner_data = self.extract_banner_data(banner_element, url)
                if banner_data:
                    banner_data_list.append(banner_data)
            
            elapsed = time.time() - start_time
            
            if banner_data_list:
                if self.debug_mode:
                    total_buttons = sum(len(banner['buttons']) for banner in banner_data_list)
                    logger.info(f"✅ SUCCESS: {len(banner_data_list)} banners, {total_buttons} buttons ({elapsed:.1f}s)")
                return (url, True, banner_data_list)
            else:
                if self.debug_mode:
                    logger.info(f"ℹ️ NO BANNERS: {url} ({elapsed:.1f}s)")
                return (url, True, [])
                
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"❌ ERROR: {url} - {str(e)[:100]} ({elapsed:.1f}s)")
            return (url, False, [])
        finally:
            if driver:
                try:
                    driver.quit()
                except:
                    pass

    def collect_all_banners(self, max_urls=None):
        """Sammle alle Banner von URLs"""
        logger.info("🚀 Cookie Banner Collection starting...")
        
        urls_to_process = ALLE_NEUEN_URLS[:max_urls] if max_urls else ALLE_NEUEN_URLS
        all_banner_data = []
        
        # Sequenziell (einfacher)
        for i, url in enumerate(urls_to_process):
            url_result, success, banner_data = self.scrape_url(url)
            
            if success:
                all_banner_data.extend(banner_data)
                self.stats['successful'] += 1
                self.stats['banners'] += len(banner_data)
            else:
                self.failed_urls.append(url)
            
            self.stats['processed'] += 1
            
            # Progress
            progress = ((i + 1) / len(urls_to_process)) * 100
            total_buttons = sum(len(banner['buttons']) for banner in all_banner_data)
            logger.info(f"📈 Progress: {i + 1}/{len(urls_to_process)} ({progress:.1f}%) - "
                       f"Banners: {len(all_banner_data)}, Buttons: {total_buttons}")
        
        return all_banner_data

    def save_raw_data(self, banner_data_list):
        """Speichere Roh-Daten für weitere Verarbeitung"""
        if not banner_data_list:
            logger.warning("⚠️ No banner data to save.")
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Speichere als JSON (für nächstes Script)
        json_file = f"raw_banner_data_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(banner_data_list, f, indent=2, ensure_ascii=False, default=str)
        
        # Statistiken
        total_buttons = sum(len(banner['buttons']) for banner in banner_data_list)
        domains = set(banner['domain'] for banner in banner_data_list)
        
        logger.info(f"💾 Raw data saved: {json_file}")
        logger.info(f"📊 Statistics:")
        logger.info(f"   Banners collected: {len(banner_data_list)}")
        logger.info(f"   Total buttons: {total_buttons}")
        logger.info(f"   Unique domains: {len(domains)}")
        logger.info(f"   Success rate: {self.stats['successful']}/{self.stats['processed']} URLs")
        
        return json_file

def main():
    """Hauptfunktion - Sammle Banner-Daten"""
    print("🚀 SCRIPT 1: Cookie Banner Scraper")
    print("🎯 Sammelt komplette Banner mit allen Buttons")
    print("-" * 50)
    
    scraper = CookieBannerScraper()
    
    try:
        # Sammle Banner (optional: max_urls=50 für Test)
        banner_data = scraper.collect_all_banners(max_urls=100)  # Test mit 100 URLs
        
        if banner_data:
            json_file = scraper.save_raw_data(banner_data)
            print(f"\n🎉 COLLECTION COMPLETED!")
            print(f"📁 Raw data saved to: {json_file}")
            print(f"▶️ Next: Run Script 2 to label the data")
        else:
            print("❌ No banner data collected.")
            
    except KeyboardInterrupt:
        print("\n⏹️ Collection stopped by user.")
    except Exception as e:
        print(f"💥 Error: {e}")

if __name__ == "__main__":
    main(),
            r'^\s*

    def detect_cmp_type(self, element):
        """Erkenne CMP-Anbieter"""
        try:
            class_name = element.get_attribute('class') or ''
            id_attr = element.get_attribute('id') or ''
            combined = f"{class_name} {id_attr}".lower()
            
            if 'onetrust' in combined:
                return 'OneTrust'
            elif 'cookiebot' in combined or 'cybot' in combined:
                return 'Cookiebot'
            elif 'usercentrics' in combined:
                return 'Usercentrics'
            elif 'didomi' in combined:
                return 'Didomi'
            elif 'sp-message' in combined or 'sourcepoint' in combined:
                return 'Sourcepoint'
            elif 'cmp' in combined:
                return 'Consentmanager'
            elif 'borlabs' in combined:
                return 'Borlabs'
            else:
                return 'Unknown'
        except:
            return 'Unknown'

    def scrape_url(self, url):
        """Scrape einzelne URL"""
        driver = None
        start_time = time.time()
        
        try:
            if self.debug_mode:
                logger.info(f"🌐 Processing: {url}")
            
            driver = self.setup_driver()
            driver.get(url)
            
            # Warte auf Seite
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            time.sleep(5)  # Banner-Ladezeit
            
            # Suche Banner
            banners = self.find_cookie_banners(driver)
            
            # Wenn keine: Trigger
            if not banners:
                self.trigger_banners(driver)
                banners = self.find_cookie_banners(driver)
            
            # Extrahiere Banner-Daten
            banner_data_list = []
            for banner_element in banners:
                banner_data = self.extract_banner_data(banner_element, url)
                if banner_data:
                    banner_data_list.append(banner_data)
            
            elapsed = time.time() - start_time
            
            if banner_data_list:
                if self.debug_mode:
                    total_buttons = sum(len(banner['buttons']) for banner in banner_data_list)
                    logger.info(f"✅ SUCCESS: {len(banner_data_list)} banners, {total_buttons} buttons ({elapsed:.1f}s)")
                return (url, True, banner_data_list)
            else:
                if self.debug_mode:
                    logger.info(f"ℹ️ NO BANNERS: {url} ({elapsed:.1f}s)")
                return (url, True, [])
                
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"❌ ERROR: {url} - {str(e)[:100]} ({elapsed:.1f}s)")
            return (url, False, [])
        finally:
            if driver:
                try:
                    driver.quit()
                except:
                    pass

    def collect_all_banners(self, max_urls=None):
        """Sammle alle Banner von URLs"""
        logger.info("🚀 Cookie Banner Collection starting...")
        
        urls_to_process = ALLE_NEUEN_URLS[:max_urls] if max_urls else ALLE_NEUEN_URLS
        all_banner_data = []
        
        # Sequenziell (einfacher)
        for i, url in enumerate(urls_to_process):
            url_result, success, banner_data = self.scrape_url(url)
            
            if success:
                all_banner_data.extend(banner_data)
                self.stats['successful'] += 1
                self.stats['banners'] += len(banner_data)
            else:
                self.failed_urls.append(url)
            
            self.stats['processed'] += 1
            
            # Progress
            progress = ((i + 1) / len(urls_to_process)) * 100
            total_buttons = sum(len(banner['buttons']) for banner in all_banner_data)
            logger.info(f"📈 Progress: {i + 1}/{len(urls_to_process)} ({progress:.1f}%) - "
                       f"Banners: {len(all_banner_data)}, Buttons: {total_buttons}")
        
        return all_banner_data

    def save_raw_data(self, banner_data_list):
        """Speichere Roh-Daten für weitere Verarbeitung"""
        if not banner_data_list:
            logger.warning("⚠️ No banner data to save.")
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Speichere als JSON (für nächstes Script)
        json_file = f"raw_banner_data_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(banner_data_list, f, indent=2, ensure_ascii=False, default=str)
        
        # Statistiken
        total_buttons = sum(len(banner['buttons']) for banner in banner_data_list)
        domains = set(banner['domain'] for banner in banner_data_list)
        
        logger.info(f"💾 Raw data saved: {json_file}")
        logger.info(f"📊 Statistics:")
        logger.info(f"   Banners collected: {len(banner_data_list)}")
        logger.info(f"   Total buttons: {total_buttons}")
        logger.info(f"   Unique domains: {len(domains)}")
        logger.info(f"   Success rate: {self.stats['successful']}/{self.stats['processed']} URLs")
        
        return json_file

def main():
    """Hauptfunktion - Sammle Banner-Daten"""
    print("🚀 SCRIPT 1: Cookie Banner Scraper")
    print("🎯 Sammelt komplette Banner mit allen Buttons")
    print("-" * 50)
    
    scraper = CookieBannerScraper()
    
    try:
        # Sammle Banner (optional: max_urls=50 für Test)
        banner_data = scraper.collect_all_banners(max_urls=100)  # Test mit 100 URLs
        
        if banner_data:
            json_file = scraper.save_raw_data(banner_data)
            print(f"\n🎉 COLLECTION COMPLETED!")
            print(f"📁 Raw data saved to: {json_file}")
            print(f"▶️ Next: Run Script 2 to label the data")
        else:
            print("❌ No banner data collected.")
            
    except KeyboardInterrupt:
        print("\n⏹️ Collection stopped by user.")
    except Exception as e:
        print(f"💥 Error: {e}")

if __name__ == "__main__":
    main()  # Nur Whitespace
        ]
        
        for pattern in exclusion_patterns:
            if re.search(pattern, text_lower):
                if self.debug_mode:
                    logger.debug(f"Button text excluded by pattern '{pattern}': '{text}'")
                return False
        
        # STARK ERWEITERTE Cookie-Keywords (wie dein Original)
        cookie_keywords = [
            # Basis Cookie-Begriffe
            'cookie', 'cookies', 'consent', 'einwilligung', 'zustimm', 'akzeptier', 'ablehnen',
            'verweiger', 'erlauben', 'zulassen', 'einstellung', 'präferenz', 'datenschutz',
            'privacy', 'accept', 'decline', 'reject', 'allow', 'deny', 'agree',
            'disagree', 'opt-in', 'opt-out', 'manage', 'customize', 'settings',
            
            # Funktionale Begriffe
            'notwendig', 'erforderlich', 'essenziell', 'funktional', 'marketing',
            'analytics', 'tracking', 'personalisier', 'werbung', 'statistik',
            'essential', 'necessary', 'required', 'functional', 'advertising',
            
            # Quantifizierer
            'alle', 'all', 'nur', 'only', 'basic', 'advanced', 'erweitert',
            
            # Aktionen
            'speichern', 'save', 'bestätigen', 'confirm', 'fortfahren', 'continue',
            'verstanden', 'geht klar', 'in ordnung', 'einverstanden', 'ok',
            'werbefrei', 'werbefinanziert', 'auswählen', 'wählen', 'bestätigung',
            
            # Deutsche Phrasen
            'diese website', 'ihre daten', 'cookies verwenden', 'datenverarbeitung',
            'datenschutzerklärung', 'cookies und', 'partner', 'analyse',
            
            # Einfache Zustimmungs-/Ablehnungswörter
            'ja', 'yes', 'nein', 'no', 'okay', 'got it', 'verstehe', 'fine',
            'sure', 'weiter', 'back', 'zurück', 'mehr', 'more', 'info',
            'details', 'learn', 'erfahren'
        ]
        
        has_cookie_keyword = any(keyword in text_lower for keyword in cookie_keywords)
        if not has_cookie_keyword:
            if self.debug_mode:
                logger.debug(f"No cookie keywords found in button: '{text}'")
            return False
            
        return True

    def detect_cmp_type(self, element):
        """Erkenne CMP-Anbieter"""
        try:
            class_name = element.get_attribute('class') or ''
            id_attr = element.get_attribute('id') or ''
            combined = f"{class_name} {id_attr}".lower()
            
            if 'onetrust' in combined:
                return 'OneTrust'
            elif 'cookiebot' in combined or 'cybot' in combined:
                return 'Cookiebot'
            elif 'usercentrics' in combined:
                return 'Usercentrics'
            elif 'didomi' in combined:
                return 'Didomi'
            elif 'sp-message' in combined or 'sourcepoint' in combined:
                return 'Sourcepoint'
            elif 'cmp' in combined:
                return 'Consentmanager'
            elif 'borlabs' in combined:
                return 'Borlabs'
            else:
                return 'Unknown'
        except:
            return 'Unknown'

    def scrape_url(self, url):
        """Scrape einzelne URL"""
        driver = None
        start_time = time.time()
        
        try:
            if self.debug_mode:
                logger.info(f"🌐 Processing: {url}")
            
            driver = self.setup_driver()
            driver.get(url)
            
            # Warte auf Seite
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            time.sleep(5)  # Banner-Ladezeit
            
            # Suche Banner
            banners = self.find_cookie_banners(driver)
            
            # Wenn keine: Trigger
            if not banners:
                self.trigger_banners(driver)
                banners = self.find_cookie_banners(driver)
            
            # Extrahiere Banner-Daten
            banner_data_list = []
            for banner_element in banners:
                banner_data = self.extract_banner_data(banner_element, url)
                if banner_data:
                    banner_data_list.append(banner_data)
            
            elapsed = time.time() - start_time
            
            if banner_data_list:
                if self.debug_mode:
                    total_buttons = sum(len(banner['buttons']) for banner in banner_data_list)
                    logger.info(f"✅ SUCCESS: {len(banner_data_list)} banners, {total_buttons} buttons ({elapsed:.1f}s)")
                return (url, True, banner_data_list)
            else:
                if self.debug_mode:
                    logger.info(f"ℹ️ NO BANNERS: {url} ({elapsed:.1f}s)")
                return (url, True, [])
                
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"❌ ERROR: {url} - {str(e)[:100]} ({elapsed:.1f}s)")
            return (url, False, [])
        finally:
            if driver:
                try:
                    driver.quit()
                except:
                    pass

    def collect_all_banners(self, max_urls=None):
        """Sammle alle Banner von URLs"""
        logger.info("🚀 Cookie Banner Collection starting...")
        
        urls_to_process = ALLE_NEUEN_URLS[:max_urls] if max_urls else ALLE_NEUEN_URLS
        all_banner_data = []
        
        # Sequenziell (einfacher)
        for i, url in enumerate(urls_to_process):
            url_result, success, banner_data = self.scrape_url(url)
            
            if success:
                all_banner_data.extend(banner_data)
                self.stats['successful'] += 1
                self.stats['banners'] += len(banner_data)
            else:
                self.failed_urls.append(url)
            
            self.stats['processed'] += 1
            
            # Progress
            progress = ((i + 1) / len(urls_to_process)) * 100
            total_buttons = sum(len(banner['buttons']) for banner in all_banner_data)
            logger.info(f"📈 Progress: {i + 1}/{len(urls_to_process)} ({progress:.1f}%) - "
                       f"Banners: {len(all_banner_data)}, Buttons: {total_buttons}")
        
        return all_banner_data

    def save_raw_data(self, banner_data_list):
        """Speichere Roh-Daten für weitere Verarbeitung"""
        if not banner_data_list:
            logger.warning("⚠️ No banner data to save.")
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Speichere als JSON (für nächstes Script)
        json_file = f"raw_banner_data_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(banner_data_list, f, indent=2, ensure_ascii=False, default=str)
        
        # Statistiken
        total_buttons = sum(len(banner['buttons']) for banner in banner_data_list)
        domains = set(banner['domain'] for banner in banner_data_list)
        
        logger.info(f"💾 Raw data saved: {json_file}")
        logger.info(f"📊 Statistics:")
        logger.info(f"   Banners collected: {len(banner_data_list)}")
        logger.info(f"   Total buttons: {total_buttons}")
        logger.info(f"   Unique domains: {len(domains)}")
        logger.info(f"   Success rate: {self.stats['successful']}/{self.stats['processed']} URLs")
        
        return json_file

def main():
    """Hauptfunktion - Sammle Banner-Daten"""
    print("🚀 SCRIPT 1: Cookie Banner Scraper")
    print("🎯 Sammelt komplette Banner mit allen Buttons")
    print("-" * 50)
    
    scraper = CookieBannerScraper()
    
    try:
        # Sammle Banner (optional: max_urls=50 für Test)
        banner_data = scraper.collect_all_banners(max_urls=100)  # Test mit 100 URLs
        
        if banner_data:
            json_file = scraper.save_raw_data(banner_data)
            print(f"\n🎉 COLLECTION COMPLETED!")
            print(f"📁 Raw data saved to: {json_file}")
            print(f"▶️ Next: Run Script 2 to label the data")
        else:
            print("❌ No banner data collected.")
            
    except KeyboardInterrupt:
        print("\n⏹️ Collection stopped by user.")
    except Exception as e:
        print(f"💥 Error: {e}")

if __name__ == "__main__":
    main()