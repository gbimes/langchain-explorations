"""Generated client library for recaptchaenterprise version v1."""
# NOTE: This file is autogenerated and should not be edited by hand.

from __future__ import absolute_import

from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.recaptchaenterprise.v1 import recaptchaenterprise_v1_messages as messages


class RecaptchaenterpriseV1(base_api.BaseApiClient):
  """Generated client library for service recaptchaenterprise version v1."""

  MESSAGES_MODULE = messages
  BASE_URL = 'https://recaptchaenterprise.googleapis.com/'
  MTLS_BASE_URL = 'https://recaptchaenterprise.mtls.googleapis.com/'

  _PACKAGE = 'recaptchaenterprise'
  _SCOPES = ['https://www.googleapis.com/auth/cloud-platform']
  _VERSION = 'v1'
  _CLIENT_ID = 'CLIENT_ID'
  _CLIENT_SECRET = 'CLIENT_SECRET'
  _USER_AGENT = 'google-cloud-sdk'
  _CLIENT_CLASS_NAME = 'RecaptchaenterpriseV1'
  _URL_VERSION = 'v1'
  _API_KEY = None

  def __init__(self, url='', credentials=None,
               get_credentials=True, http=None, model=None,
               log_request=False, log_response=False,
               credentials_args=None, default_global_params=None,
               additional_http_headers=None, response_encoding=None):
    """Create a new recaptchaenterprise handle."""
    url = url or self.BASE_URL
    super(RecaptchaenterpriseV1, self).__init__(
        url, credentials=credentials,
        get_credentials=get_credentials, http=http, model=model,
        log_request=log_request, log_response=log_response,
        credentials_args=credentials_args,
        default_global_params=default_global_params,
        additional_http_headers=additional_http_headers,
        response_encoding=response_encoding)
    self.projects_assessments = self.ProjectsAssessmentsService(self)
    self.projects_firewallpolicies = self.ProjectsFirewallpoliciesService(self)
    self.projects_keys = self.ProjectsKeysService(self)
    self.projects_relatedaccountgroupmemberships = self.ProjectsRelatedaccountgroupmembershipsService(self)
    self.projects_relatedaccountgroups_memberships = self.ProjectsRelatedaccountgroupsMembershipsService(self)
    self.projects_relatedaccountgroups = self.ProjectsRelatedaccountgroupsService(self)
    self.projects = self.ProjectsService(self)

  class ProjectsAssessmentsService(base_api.BaseApiService):
    """Service class for the projects_assessments resource."""

    _NAME = 'projects_assessments'

    def __init__(self, client):
      super(RecaptchaenterpriseV1.ProjectsAssessmentsService, self).__init__(client)
      self._upload_configs = {
          }

    def Annotate(self, request, global_params=None):
      r"""Annotates a previously created Assessment to provide additional information on whether the event turned out to be authentic or fraudulent.

      Args:
        request: (RecaptchaenterpriseProjectsAssessmentsAnnotateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRecaptchaenterpriseV1AnnotateAssessmentResponse) The response message.
      """
      config = self.GetMethodConfig('Annotate')
      return self._RunMethod(
          config, request, global_params=global_params)

    Annotate.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1/projects/{projectsId}/assessments/{assessmentsId}:annotate',
        http_method='POST',
        method_id='recaptchaenterprise.projects.assessments.annotate',
        ordered_params=['name'],
        path_params=['name'],
        query_params=[],
        relative_path='v1/{+name}:annotate',
        request_field='googleCloudRecaptchaenterpriseV1AnnotateAssessmentRequest',
        request_type_name='RecaptchaenterpriseProjectsAssessmentsAnnotateRequest',
        response_type_name='GoogleCloudRecaptchaenterpriseV1AnnotateAssessmentResponse',
        supports_download=False,
    )

    def Create(self, request, global_params=None):
      r"""Creates an Assessment of the likelihood an event is legitimate.

      Args:
        request: (RecaptchaenterpriseProjectsAssessmentsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRecaptchaenterpriseV1Assessment) The response message.
      """
      config = self.GetMethodConfig('Create')
      return self._RunMethod(
          config, request, global_params=global_params)

    Create.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1/projects/{projectsId}/assessments',
        http_method='POST',
        method_id='recaptchaenterprise.projects.assessments.create',
        ordered_params=['parent'],
        path_params=['parent'],
        query_params=[],
        relative_path='v1/{+parent}/assessments',
        request_field='googleCloudRecaptchaenterpriseV1Assessment',
        request_type_name='RecaptchaenterpriseProjectsAssessmentsCreateRequest',
        response_type_name='GoogleCloudRecaptchaenterpriseV1Assessment',
        supports_download=False,
    )

  class ProjectsFirewallpoliciesService(base_api.BaseApiService):
    """Service class for the projects_firewallpolicies resource."""

    _NAME = 'projects_firewallpolicies'

    def __init__(self, client):
      super(RecaptchaenterpriseV1.ProjectsFirewallpoliciesService, self).__init__(client)
      self._upload_configs = {
          }

    def Create(self, request, global_params=None):
      r"""Creates a new FirewallPolicy, specifying conditions at which reCAPTCHA Enterprise actions can be executed. A project may have a maximum of 1000 policies.

      Args:
        request: (RecaptchaenterpriseProjectsFirewallpoliciesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRecaptchaenterpriseV1FirewallPolicy) The response message.
      """
      config = self.GetMethodConfig('Create')
      return self._RunMethod(
          config, request, global_params=global_params)

    Create.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1/projects/{projectsId}/firewallpolicies',
        http_method='POST',
        method_id='recaptchaenterprise.projects.firewallpolicies.create',
        ordered_params=['parent'],
        path_params=['parent'],
        query_params=[],
        relative_path='v1/{+parent}/firewallpolicies',
        request_field='googleCloudRecaptchaenterpriseV1FirewallPolicy',
        request_type_name='RecaptchaenterpriseProjectsFirewallpoliciesCreateRequest',
        response_type_name='GoogleCloudRecaptchaenterpriseV1FirewallPolicy',
        supports_download=False,
    )

    def Delete(self, request, global_params=None):
      r"""Deletes the specified firewall policy.

      Args:
        request: (RecaptchaenterpriseProjectsFirewallpoliciesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
      config = self.GetMethodConfig('Delete')
      return self._RunMethod(
          config, request, global_params=global_params)

    Delete.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1/projects/{projectsId}/firewallpolicies/{firewallpoliciesId}',
        http_method='DELETE',
        method_id='recaptchaenterprise.projects.firewallpolicies.delete',
        ordered_params=['name'],
        path_params=['name'],
        query_params=[],
        relative_path='v1/{+name}',
        request_field='',
        request_type_name='RecaptchaenterpriseProjectsFirewallpoliciesDeleteRequest',
        response_type_name='GoogleProtobufEmpty',
        supports_download=False,
    )

    def Get(self, request, global_params=None):
      r"""Returns the specified firewall policy.

      Args:
        request: (RecaptchaenterpriseProjectsFirewallpoliciesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRecaptchaenterpriseV1FirewallPolicy) The response message.
      """
      config = self.GetMethodConfig('Get')
      return self._RunMethod(
          config, request, global_params=global_params)

    Get.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1/projects/{projectsId}/firewallpolicies/{firewallpoliciesId}',
        http_method='GET',
        method_id='recaptchaenterprise.projects.firewallpolicies.get',
        ordered_params=['name'],
        path_params=['name'],
        query_params=[],
        relative_path='v1/{+name}',
        request_field='',
        request_type_name='RecaptchaenterpriseProjectsFirewallpoliciesGetRequest',
        response_type_name='GoogleCloudRecaptchaenterpriseV1FirewallPolicy',
        supports_download=False,
    )

    def List(self, request, global_params=None):
      r"""Returns the list of all firewall policies that belong to a project.

      Args:
        request: (RecaptchaenterpriseProjectsFirewallpoliciesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRecaptchaenterpriseV1ListFirewallPoliciesResponse) The response message.
      """
      config = self.GetMethodConfig('List')
      return self._RunMethod(
          config, request, global_params=global_params)

    List.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1/projects/{projectsId}/firewallpolicies',
        http_method='GET',
        method_id='recaptchaenterprise.projects.firewallpolicies.list',
        ordered_params=['parent'],
        path_params=['parent'],
        query_params=['pageSize', 'pageToken'],
        relative_path='v1/{+parent}/firewallpolicies',
        request_field='',
        request_type_name='RecaptchaenterpriseProjectsFirewallpoliciesListRequest',
        response_type_name='GoogleCloudRecaptchaenterpriseV1ListFirewallPoliciesResponse',
        supports_download=False,
    )

    def Patch(self, request, global_params=None):
      r"""Updates the specified firewall policy.

      Args:
        request: (RecaptchaenterpriseProjectsFirewallpoliciesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRecaptchaenterpriseV1FirewallPolicy) The response message.
      """
      config = self.GetMethodConfig('Patch')
      return self._RunMethod(
          config, request, global_params=global_params)

    Patch.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1/projects/{projectsId}/firewallpolicies/{firewallpoliciesId}',
        http_method='PATCH',
        method_id='recaptchaenterprise.projects.firewallpolicies.patch',
        ordered_params=['name'],
        path_params=['name'],
        query_params=['updateMask'],
        relative_path='v1/{+name}',
        request_field='googleCloudRecaptchaenterpriseV1FirewallPolicy',
        request_type_name='RecaptchaenterpriseProjectsFirewallpoliciesPatchRequest',
        response_type_name='GoogleCloudRecaptchaenterpriseV1FirewallPolicy',
        supports_download=False,
    )

    def Reorder(self, request, global_params=None):
      r"""Reorders all firewall policies.

      Args:
        request: (RecaptchaenterpriseProjectsFirewallpoliciesReorderRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRecaptchaenterpriseV1ReorderFirewallPoliciesResponse) The response message.
      """
      config = self.GetMethodConfig('Reorder')
      return self._RunMethod(
          config, request, global_params=global_params)

    Reorder.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1/projects/{projectsId}/firewallpolicies:reorder',
        http_method='POST',
        method_id='recaptchaenterprise.projects.firewallpolicies.reorder',
        ordered_params=['parent'],
        path_params=['parent'],
        query_params=[],
        relative_path='v1/{+parent}/firewallpolicies:reorder',
        request_field='googleCloudRecaptchaenterpriseV1ReorderFirewallPoliciesRequest',
        request_type_name='RecaptchaenterpriseProjectsFirewallpoliciesReorderRequest',
        response_type_name='GoogleCloudRecaptchaenterpriseV1ReorderFirewallPoliciesResponse',
        supports_download=False,
    )

  class ProjectsKeysService(base_api.BaseApiService):
    """Service class for the projects_keys resource."""

    _NAME = 'projects_keys'

    def __init__(self, client):
      super(RecaptchaenterpriseV1.ProjectsKeysService, self).__init__(client)
      self._upload_configs = {
          }

    def AddIpOverride(self, request, global_params=None):
      r"""Adds an IP override to a key. The following restrictions hold: * The maximum number of IP overrides per key is 100. * For any conflict (such as IP already exists or IP part of an existing IP range), an error is returned.

      Args:
        request: (RecaptchaenterpriseProjectsKeysAddIpOverrideRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRecaptchaenterpriseV1AddIpOverrideResponse) The response message.
      """
      config = self.GetMethodConfig('AddIpOverride')
      return self._RunMethod(
          config, request, global_params=global_params)

    AddIpOverride.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1/projects/{projectsId}/keys/{keysId}:addIpOverride',
        http_method='POST',
        method_id='recaptchaenterprise.projects.keys.addIpOverride',
        ordered_params=['name'],
        path_params=['name'],
        query_params=[],
        relative_path='v1/{+name}:addIpOverride',
        request_field='googleCloudRecaptchaenterpriseV1AddIpOverrideRequest',
        request_type_name='RecaptchaenterpriseProjectsKeysAddIpOverrideRequest',
        response_type_name='GoogleCloudRecaptchaenterpriseV1AddIpOverrideResponse',
        supports_download=False,
    )

    def Create(self, request, global_params=None):
      r"""Creates a new reCAPTCHA Enterprise key.

      Args:
        request: (RecaptchaenterpriseProjectsKeysCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRecaptchaenterpriseV1Key) The response message.
      """
      config = self.GetMethodConfig('Create')
      return self._RunMethod(
          config, request, global_params=global_params)

    Create.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1/projects/{projectsId}/keys',
        http_method='POST',
        method_id='recaptchaenterprise.projects.keys.create',
        ordered_params=['parent'],
        path_params=['parent'],
        query_params=[],
        relative_path='v1/{+parent}/keys',
        request_field='googleCloudRecaptchaenterpriseV1Key',
        request_type_name='RecaptchaenterpriseProjectsKeysCreateRequest',
        response_type_name='GoogleCloudRecaptchaenterpriseV1Key',
        supports_download=False,
    )

    def Delete(self, request, global_params=None):
      r"""Deletes the specified key.

      Args:
        request: (RecaptchaenterpriseProjectsKeysDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
      config = self.GetMethodConfig('Delete')
      return self._RunMethod(
          config, request, global_params=global_params)

    Delete.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1/projects/{projectsId}/keys/{keysId}',
        http_method='DELETE',
        method_id='recaptchaenterprise.projects.keys.delete',
        ordered_params=['name'],
        path_params=['name'],
        query_params=[],
        relative_path='v1/{+name}',
        request_field='',
        request_type_name='RecaptchaenterpriseProjectsKeysDeleteRequest',
        response_type_name='GoogleProtobufEmpty',
        supports_download=False,
    )

    def Get(self, request, global_params=None):
      r"""Returns the specified key.

      Args:
        request: (RecaptchaenterpriseProjectsKeysGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRecaptchaenterpriseV1Key) The response message.
      """
      config = self.GetMethodConfig('Get')
      return self._RunMethod(
          config, request, global_params=global_params)

    Get.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1/projects/{projectsId}/keys/{keysId}',
        http_method='GET',
        method_id='recaptchaenterprise.projects.keys.get',
        ordered_params=['name'],
        path_params=['name'],
        query_params=[],
        relative_path='v1/{+name}',
        request_field='',
        request_type_name='RecaptchaenterpriseProjectsKeysGetRequest',
        response_type_name='GoogleCloudRecaptchaenterpriseV1Key',
        supports_download=False,
    )

    def GetMetrics(self, request, global_params=None):
      r"""Get some aggregated metrics for a Key. This data can be used to build dashboards.

      Args:
        request: (RecaptchaenterpriseProjectsKeysGetMetricsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRecaptchaenterpriseV1Metrics) The response message.
      """
      config = self.GetMethodConfig('GetMetrics')
      return self._RunMethod(
          config, request, global_params=global_params)

    GetMetrics.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1/projects/{projectsId}/keys/{keysId}/metrics',
        http_method='GET',
        method_id='recaptchaenterprise.projects.keys.getMetrics',
        ordered_params=['name'],
        path_params=['name'],
        query_params=[],
        relative_path='v1/{+name}',
        request_field='',
        request_type_name='RecaptchaenterpriseProjectsKeysGetMetricsRequest',
        response_type_name='GoogleCloudRecaptchaenterpriseV1Metrics',
        supports_download=False,
    )

    def List(self, request, global_params=None):
      r"""Returns the list of all keys that belong to a project.

      Args:
        request: (RecaptchaenterpriseProjectsKeysListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRecaptchaenterpriseV1ListKeysResponse) The response message.
      """
      config = self.GetMethodConfig('List')
      return self._RunMethod(
          config, request, global_params=global_params)

    List.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1/projects/{projectsId}/keys',
        http_method='GET',
        method_id='recaptchaenterprise.projects.keys.list',
        ordered_params=['parent'],
        path_params=['parent'],
        query_params=['pageSize', 'pageToken'],
        relative_path='v1/{+parent}/keys',
        request_field='',
        request_type_name='RecaptchaenterpriseProjectsKeysListRequest',
        response_type_name='GoogleCloudRecaptchaenterpriseV1ListKeysResponse',
        supports_download=False,
    )

    def ListIpOverrides(self, request, global_params=None):
      r"""Lists all IP overrides for a key.

      Args:
        request: (RecaptchaenterpriseProjectsKeysListIpOverridesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRecaptchaenterpriseV1ListIpOverridesResponse) The response message.
      """
      config = self.GetMethodConfig('ListIpOverrides')
      return self._RunMethod(
          config, request, global_params=global_params)

    ListIpOverrides.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1/projects/{projectsId}/keys/{keysId}:listIpOverrides',
        http_method='GET',
        method_id='recaptchaenterprise.projects.keys.listIpOverrides',
        ordered_params=['parent'],
        path_params=['parent'],
        query_params=['pageSize', 'pageToken'],
        relative_path='v1/{+parent}:listIpOverrides',
        request_field='',
        request_type_name='RecaptchaenterpriseProjectsKeysListIpOverridesRequest',
        response_type_name='GoogleCloudRecaptchaenterpriseV1ListIpOverridesResponse',
        supports_download=False,
    )

    def Migrate(self, request, global_params=None):
      r"""Migrates an existing key from reCAPTCHA to reCAPTCHA Enterprise. Once a key is migrated, it can be used from either product. SiteVerify requests are billed as CreateAssessment calls. You must be authenticated as one of the current owners of the reCAPTCHA Key, and your user must have the reCAPTCHA Enterprise Admin IAM role in the destination project.

      Args:
        request: (RecaptchaenterpriseProjectsKeysMigrateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRecaptchaenterpriseV1Key) The response message.
      """
      config = self.GetMethodConfig('Migrate')
      return self._RunMethod(
          config, request, global_params=global_params)

    Migrate.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1/projects/{projectsId}/keys/{keysId}:migrate',
        http_method='POST',
        method_id='recaptchaenterprise.projects.keys.migrate',
        ordered_params=['name'],
        path_params=['name'],
        query_params=[],
        relative_path='v1/{+name}:migrate',
        request_field='googleCloudRecaptchaenterpriseV1MigrateKeyRequest',
        request_type_name='RecaptchaenterpriseProjectsKeysMigrateRequest',
        response_type_name='GoogleCloudRecaptchaenterpriseV1Key',
        supports_download=False,
    )

    def Patch(self, request, global_params=None):
      r"""Updates the specified key.

      Args:
        request: (RecaptchaenterpriseProjectsKeysPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRecaptchaenterpriseV1Key) The response message.
      """
      config = self.GetMethodConfig('Patch')
      return self._RunMethod(
          config, request, global_params=global_params)

    Patch.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1/projects/{projectsId}/keys/{keysId}',
        http_method='PATCH',
        method_id='recaptchaenterprise.projects.keys.patch',
        ordered_params=['name'],
        path_params=['name'],
        query_params=['updateMask'],
        relative_path='v1/{+name}',
        request_field='googleCloudRecaptchaenterpriseV1Key',
        request_type_name='RecaptchaenterpriseProjectsKeysPatchRequest',
        response_type_name='GoogleCloudRecaptchaenterpriseV1Key',
        supports_download=False,
    )

    def RemoveIpOverride(self, request, global_params=None):
      r"""Removes an IP override from a key. The following restrictions hold: * If the IP isn't found in an existing IP override, a `NOT_FOUND` error is returned. * If the IP is found in an existing IP override, but the override type does not match, a `NOT_FOUND` error is returned.

      Args:
        request: (RecaptchaenterpriseProjectsKeysRemoveIpOverrideRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRecaptchaenterpriseV1RemoveIpOverrideResponse) The response message.
      """
      config = self.GetMethodConfig('RemoveIpOverride')
      return self._RunMethod(
          config, request, global_params=global_params)

    RemoveIpOverride.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1/projects/{projectsId}/keys/{keysId}:removeIpOverride',
        http_method='POST',
        method_id='recaptchaenterprise.projects.keys.removeIpOverride',
        ordered_params=['name'],
        path_params=['name'],
        query_params=[],
        relative_path='v1/{+name}:removeIpOverride',
        request_field='googleCloudRecaptchaenterpriseV1RemoveIpOverrideRequest',
        request_type_name='RecaptchaenterpriseProjectsKeysRemoveIpOverrideRequest',
        response_type_name='GoogleCloudRecaptchaenterpriseV1RemoveIpOverrideResponse',
        supports_download=False,
    )

    def RetrieveLegacySecretKey(self, request, global_params=None):
      r"""Returns the secret key related to the specified public key. You must use the legacy secret key only in a 3rd party integration with legacy reCAPTCHA.

      Args:
        request: (RecaptchaenterpriseProjectsKeysRetrieveLegacySecretKeyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRecaptchaenterpriseV1RetrieveLegacySecretKeyResponse) The response message.
      """
      config = self.GetMethodConfig('RetrieveLegacySecretKey')
      return self._RunMethod(
          config, request, global_params=global_params)

    RetrieveLegacySecretKey.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1/projects/{projectsId}/keys/{keysId}:retrieveLegacySecretKey',
        http_method='GET',
        method_id='recaptchaenterprise.projects.keys.retrieveLegacySecretKey',
        ordered_params=['key'],
        path_params=['key'],
        query_params=[],
        relative_path='v1/{+key}:retrieveLegacySecretKey',
        request_field='',
        request_type_name='RecaptchaenterpriseProjectsKeysRetrieveLegacySecretKeyRequest',
        response_type_name='GoogleCloudRecaptchaenterpriseV1RetrieveLegacySecretKeyResponse',
        supports_download=False,
    )

  class ProjectsRelatedaccountgroupmembershipsService(base_api.BaseApiService):
    """Service class for the projects_relatedaccountgroupmemberships resource."""

    _NAME = 'projects_relatedaccountgroupmemberships'

    def __init__(self, client):
      super(RecaptchaenterpriseV1.ProjectsRelatedaccountgroupmembershipsService, self).__init__(client)
      self._upload_configs = {
          }

    def Search(self, request, global_params=None):
      r"""Search group memberships related to a given account.

      Args:
        request: (RecaptchaenterpriseProjectsRelatedaccountgroupmembershipsSearchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRecaptchaenterpriseV1SearchRelatedAccountGroupMembershipsResponse) The response message.
      """
      config = self.GetMethodConfig('Search')
      return self._RunMethod(
          config, request, global_params=global_params)

    Search.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1/projects/{projectsId}/relatedaccountgroupmemberships:search',
        http_method='POST',
        method_id='recaptchaenterprise.projects.relatedaccountgroupmemberships.search',
        ordered_params=['project'],
        path_params=['project'],
        query_params=[],
        relative_path='v1/{+project}/relatedaccountgroupmemberships:search',
        request_field='googleCloudRecaptchaenterpriseV1SearchRelatedAccountGroupMembershipsRequest',
        request_type_name='RecaptchaenterpriseProjectsRelatedaccountgroupmembershipsSearchRequest',
        response_type_name='GoogleCloudRecaptchaenterpriseV1SearchRelatedAccountGroupMembershipsResponse',
        supports_download=False,
    )

  class ProjectsRelatedaccountgroupsMembershipsService(base_api.BaseApiService):
    """Service class for the projects_relatedaccountgroups_memberships resource."""

    _NAME = 'projects_relatedaccountgroups_memberships'

    def __init__(self, client):
      super(RecaptchaenterpriseV1.ProjectsRelatedaccountgroupsMembershipsService, self).__init__(client)
      self._upload_configs = {
          }

    def List(self, request, global_params=None):
      r"""Get memberships in a group of related accounts.

      Args:
        request: (RecaptchaenterpriseProjectsRelatedaccountgroupsMembershipsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRecaptchaenterpriseV1ListRelatedAccountGroupMembershipsResponse) The response message.
      """
      config = self.GetMethodConfig('List')
      return self._RunMethod(
          config, request, global_params=global_params)

    List.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1/projects/{projectsId}/relatedaccountgroups/{relatedaccountgroupsId}/memberships',
        http_method='GET',
        method_id='recaptchaenterprise.projects.relatedaccountgroups.memberships.list',
        ordered_params=['parent'],
        path_params=['parent'],
        query_params=['pageSize', 'pageToken'],
        relative_path='v1/{+parent}/memberships',
        request_field='',
        request_type_name='RecaptchaenterpriseProjectsRelatedaccountgroupsMembershipsListRequest',
        response_type_name='GoogleCloudRecaptchaenterpriseV1ListRelatedAccountGroupMembershipsResponse',
        supports_download=False,
    )

  class ProjectsRelatedaccountgroupsService(base_api.BaseApiService):
    """Service class for the projects_relatedaccountgroups resource."""

    _NAME = 'projects_relatedaccountgroups'

    def __init__(self, client):
      super(RecaptchaenterpriseV1.ProjectsRelatedaccountgroupsService, self).__init__(client)
      self._upload_configs = {
          }

    def List(self, request, global_params=None):
      r"""List groups of related accounts.

      Args:
        request: (RecaptchaenterpriseProjectsRelatedaccountgroupsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRecaptchaenterpriseV1ListRelatedAccountGroupsResponse) The response message.
      """
      config = self.GetMethodConfig('List')
      return self._RunMethod(
          config, request, global_params=global_params)

    List.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1/projects/{projectsId}/relatedaccountgroups',
        http_method='GET',
        method_id='recaptchaenterprise.projects.relatedaccountgroups.list',
        ordered_params=['parent'],
        path_params=['parent'],
        query_params=['pageSize', 'pageToken'],
        relative_path='v1/{+parent}/relatedaccountgroups',
        request_field='',
        request_type_name='RecaptchaenterpriseProjectsRelatedaccountgroupsListRequest',
        response_type_name='GoogleCloudRecaptchaenterpriseV1ListRelatedAccountGroupsResponse',
        supports_download=False,
    )

  class ProjectsService(base_api.BaseApiService):
    """Service class for the projects resource."""

    _NAME = 'projects'

    def __init__(self, client):
      super(RecaptchaenterpriseV1.ProjectsService, self).__init__(client)
      self._upload_configs = {
          }
