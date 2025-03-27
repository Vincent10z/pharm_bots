from typing import Dict, Any, Optional, List
import requests
import warnings
from urllib3.exceptions import NotOpenSSLWarning

warnings.filterwarnings('ignore', category=NotOpenSSLWarning)

class PharmacyAPI:
    def __init__(self, base_url: str = "https://67e14fb758cc6bf785254550.mockapi.io/pharmacies"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def get_all_pharmacies(self) -> List[Dict[str, Any]]:
        """Fetch all pharmacies from the API"""
        try:
            response = self.session.get(self.base_url)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching pharmacies: {e}")
            return []
    
    def get_pharmacy_by_phone(self, phone_number: str) -> Optional[Dict[str, Any]]:
        """Look up a pharmacy by phone number"""
        try:
            response = self.session.get(f"{self.base_url}?phone={phone_number}")
            response.raise_for_status()
            
            try:
                result = response.json()
                if isinstance(result, str) and result == "Not found":
                    all_pharmacies = self.get_all_pharmacies()
                    for pharmacy in all_pharmacies:
                        if self._normalize_phone(pharmacy.get("phone", "")) == self._normalize_phone(phone_number):
                            return pharmacy
                    return None
                
                if isinstance(result, list) and result:
                    return result[0]
                
            except requests.JSONDecodeError:
                print("Invalid JSON response from API")
                return None
            
            all_pharmacies = self.get_all_pharmacies()
            for pharmacy in all_pharmacies:
                if self._normalize_phone(pharmacy.get("phone", "")) == self._normalize_phone(phone_number):
                    return pharmacy
            
            return None
            
        except requests.RequestException as e:
            print(f"Error fetching pharmacy by phone: {e}")
            return None

    def _normalize_phone(self, phone: str) -> str:
        """Normalize phone number for comparison"""
        return ''.join(filter(str.isdigit, phone))