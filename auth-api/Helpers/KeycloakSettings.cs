public class KeycloakSettings
{
    
    public string Host { get ; set; } = string.Empty;
    public string UserUrl { get ; set; } = string.Empty;
    public string TokenUrl { get; set; } = string.Empty;
    public string ClientId { get; set; } = string.Empty;
    public string ClientSecret { get; set; } = string.Empty;
    public string GrantType { get; set; } = "password";
    public string AdminClientId { get; set; } = string.Empty;
    public string AdminClientSecret { get; set; } = string.Empty;
    public string Authority { get; set; } = string.Empty;
    public string Audience { get; set; } = string.Empty;
    public bool RequireHttpsMetadata {get;set;} = false;
    public string RealmRolesUrl { get; set; } = string.Empty;
    public string LogoutUrl { get; set; } = string.Empty;
}
