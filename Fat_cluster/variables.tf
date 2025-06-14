variable "project" {}
variable "region"  { default = "us-central1" }
variable "zone"    { default = "us-central1-a" }

variable "public_key_path" {
  description = "Path to your local SSH public key"
}
variable "gcp_credentials_file" {
  description = "Path to the GCP service account JSON"
  type        = string
}