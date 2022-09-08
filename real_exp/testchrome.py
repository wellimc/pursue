from selenium import webdriver
#from selenium.webdriver.chrome.service import Service as ChromeService
#from webdriver_manager.chrome import ChromeDriverManager

from selenium.webdriver.firefox.service import Service as FirefoxService
from webdriver_manager.firefox import GeckoDriverManager


tempDir = "/User//Users/wellimc/pensieve/real_exp/tmp"
# no video autoplay
profile = webdriver.FirefoxProfile()
profile.set_preference("media.autoplay.enabled", True)
profile.set_preference("browser.download.dir", tempDir)


service = FirefoxService(executable_path=GeckoDriverManager().install())

driver = webdriver.Firefox(firefox_profile=profile,service=service)

url = 'http://192.168.1.13:8080/myindex_RL.html'
driver.get(url)
timeout = 5
try:
    element_present = EC.presence_of_element_located((By.ID, 'element_id'))
    WebDriverWait(driver, timeout).until(element_present)
except TimeoutException:
    print("Timed out waiting for page to load")
