# Terraform configuration for Azure Web App deployment (sreenscount-rag)

terraform {
  required_version = ">= 1.3.0"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }
}

provider "azurerm" {
  features {}
}

variable "resource_group_name" {
  type        = string
  default     = "rg-sreenscount-rag"
  description = "Name of the Azure Resource Group"
}

variable "location" {
  type        = string
  default     = "East US"
  description = "Azure Region for deployment"
}

variable "app_name" {
  type        = string
  default     = "sreenscount-rag"
  description = "Name of the Azure Web App Service"
}

variable "sku_name" {
  type        = string
  default     = "B1"
  description = "SKU for the App Service Plan (e.g. B1, P1v2)"
}

variable "pinecone_key" {
  type        = string
  sensitive   = true
  description = "Pinecone API key for vector database"
  default     = ""
}

variable "google_api_key" {
  type        = string
  sensitive   = true
  description = "Google Gemini API key for LLM recommendations and evaluation"
  default     = ""
}

resource "azurerm_resource_group" "rg" {
  name     = var.resource_group_name
  location = var.location
}

resource "azurerm_service_plan" "asp" {
  name                = "asp-${var.app_name}"
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location
  os_type             = "Linux"
  sku_name            = var.sku_name
}

resource "azurerm_linux_web_app" "webapp" {
  name                = var.app_name
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location
  service_plan_id     = azurerm_service_plan.asp.id

  site_config {
    always_on = var.sku_name == "F1" ? false : true

    application_stack {
      python_version = "3.11"
    }

    app_command_line = "uvicorn app.main:app --host 0.0.0.0 --port 8000"
  }

  app_settings = {
    "PINECONE_KEY"                   = var.pinecone_key
    "GOOGLE_API_KEY"                 = var.google_api_key
    "SCM_DO_BUILD_DURING_DEPLOYMENT" = "true"
  }
}

output "web_app_name" {
  value       = azurerm_linux_web_app.webapp.name
  description = "The name of the Azure Linux Web App"
}

output "web_app_hostname" {
  value       = azurerm_linux_web_app.webapp.default_hostname
  description = "The default hostname of the Azure Web App"
}
