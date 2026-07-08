"""Tool implementations for agent-creator role mutations."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import json
from typing import Any

from face_of_agi.agent_creator.contracts import (
    AgentCreatorToolResult,
    AgentRoleDefinition,
)
from face_of_agi.agent_creator.store import AgentCreatorStore, validate_role_definitions
from face_of_agi.models.agent_creator.contracts import (
    CreatorMutation,
    RoleAuthorInput,
    RoleAuthorModel,
)


class RoleMutationToolExecutor:
    """Execute creator-orchestrator tools against a staged role working set."""

    def __init__(
        self,
        *,
        store: AgentCreatorStore,
        role_author: RoleAuthorModel,
        run_id: int,
        roles: tuple[AgentRoleDefinition, ...],
        general_system_prompt: str,
        max_tool_calls: int,
        max_roles: int = 8,
    ) -> None:
        if max_tool_calls < 0:
            raise ValueError("agent creator max_tool_calls must be non-negative")
        if max_roles < 1:
            raise ValueError("agent creator max_roles must be at least 1")
        self.store = store
        self.role_author = role_author
        self.run_id = run_id
        self.general_system_prompt = general_system_prompt
        self.max_tool_calls = max_tool_calls
        self.max_roles = max_roles
        self._working_roles = {role.role: role for role in roles}
        self._call_count = 0

    @property
    def call_count(self) -> int:
        """Return the number of attempted tool calls."""

        return self._call_count

    @property
    def roles(self) -> tuple[AgentRoleDefinition, ...]:
        """Return the current staged working role set."""

        return tuple(self._working_roles[name] for name in sorted(self._working_roles))

    def execute_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Execute a normalized creator tool call and return JSON content."""

        self._call_count += 1
        call_index = self._call_count
        if call_index > self.max_tool_calls:
            result = AgentCreatorToolResult(
                status="failed",
                reason="max_tool_calls_exhausted",
            )
            self._record(tool_name, arguments, call_index, result)
            return _tool_result_text(result)
        try:
            result = self._execute(tool_name, arguments)
        except _ToolCallFailure as exc:
            result = AgentCreatorToolResult(status="failed", reason=exc.reason)
        except Exception as exc:
            del exc
            result = AgentCreatorToolResult(status="failed", reason="tool_error")
        self._record(tool_name, arguments, call_index, result)
        return _tool_result_text(result)

    def execute_plan(self, mutations: tuple[CreatorMutation, ...]) -> tuple[str, ...]:
        """Execute a creator mutation plan.

        Deletes are applied first so add operations can use freed role capacity.
        Role-author calls for add/update operations are then run in parallel, and
        the resulting revisions are staged sequentially.
        """

        ordered = tuple(
            item for item in mutations if item.action == "delete"
        ) + tuple(item for item in mutations if item.action != "delete")
        author_jobs: list[_AuthorJob] = []
        reserved_role_names = set(self._working_roles)
        reserved_role_count = len(self._working_roles)
        results: list[str] = []
        for mutation in ordered:
            tool_name, arguments = _mutation_tool_call(mutation)
            if tool_name == "delete":
                results.append(self.execute_tool_call(tool_name, arguments))
                reserved_role_names = set(self._working_roles)
                reserved_role_count = len(self._working_roles)
                continue
            self._call_count += 1
            call_index = self._call_count
            if call_index > self.max_tool_calls:
                result = AgentCreatorToolResult(
                    status="failed",
                    reason="max_tool_calls_exhausted",
                )
                self._record(tool_name, arguments, call_index, result)
                results.append(_tool_result_text(result))
                continue
            try:
                job = self._prepare_author_job(
                    tool_name=tool_name,
                    arguments=arguments,
                    call_index=call_index,
                    reserved_role_names=reserved_role_names,
                    reserved_role_count=reserved_role_count,
                )
            except _ToolCallFailure as exc:
                result = AgentCreatorToolResult(status="failed", reason=exc.reason)
                self._record(tool_name, arguments, call_index, result)
                results.append(_tool_result_text(result))
                continue
            if job.reserve_role_name:
                reserved_role_names.add(job.reserve_role_name)
            reserved_role_count += job.reserve_role_count
            author_jobs.append(job)

        results.extend(self._run_author_jobs(author_jobs))
        return tuple(results)

    def _execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> AgentCreatorToolResult:
        if tool_name == "delete":
            return self._delete(arguments)
        if tool_name == "add":
            return self._add(arguments)
        if tool_name == "update":
            return self._update(arguments)
        return AgentCreatorToolResult(
            status="failed",
            reason="unknown_tool",
        )

    def _delete(self, arguments: dict[str, Any]) -> AgentCreatorToolResult:
        role_name = _required_string(arguments, "role_name")
        current = self._working_roles.get(role_name)
        if current is None:
            return AgentCreatorToolResult(
                status="failed",
                reason="unknown_role",
            )
        if len(self._working_roles) <= 1:
            return AgentCreatorToolResult(
                status="failed",
                reason="final_role",
            )
        self.store.stage_role_revision(
            role=current,
            active=False,
            operation="delete",
            created_by_run_id=self.run_id,
            guidance=arguments,
        )
        del self._working_roles[role_name]
        return AgentCreatorToolResult(
            status="ok",
        )

    def _add(self, arguments: dict[str, Any]) -> AgentCreatorToolResult:
        role_name = _required_string(arguments, "role_name")
        guidance = _required_string(arguments, "instruction_guidance")
        meta_description = _required_string(arguments, "meta_description")
        if role_name in self._working_roles:
            return AgentCreatorToolResult(
                status="failed",
                reason="duplicate_role",
            )
        if len(self._working_roles) >= self.max_roles:
            return AgentCreatorToolResult(
                status="failed",
                reason="max_roles_reached",
            )
        role_instructions = self.role_author.create_role_instructions(
            RoleAuthorInput(
                role_name=role_name,
                instruction_guidance=guidance,
                general_system_prompt=self.general_system_prompt,
                metadata={"operation": "add", "run_id": self.run_id},
            )
        )
        role = AgentRoleDefinition(
            role=role_name,
            meta_description=meta_description,
            role_instructions=role_instructions,
        )
        validate_role_definitions((*self.roles, role))
        self.store.stage_role_revision(
            role=role,
            active=True,
            operation="add",
            created_by_run_id=self.run_id,
            guidance=arguments,
        )
        self._working_roles[role.role] = role
        return AgentCreatorToolResult(
            status="ok",
        )

    def _update(self, arguments: dict[str, Any]) -> AgentCreatorToolResult:
        role_name = _required_string(arguments, "role_name")
        failures = _required_string(arguments, "identified_failures")
        meta_description = _optional_string(arguments, "meta_description")
        current = self._working_roles.get(role_name)
        if current is None:
            return AgentCreatorToolResult(
                status="failed",
                reason="unknown_role",
            )
        role_instructions = self.role_author.update_role_instructions(
            RoleAuthorInput(
                role_name=role_name,
                identified_failures=failures,
                current_role=current,
                general_system_prompt=self.general_system_prompt,
                metadata={"operation": "update", "run_id": self.run_id},
            )
        )
        role = AgentRoleDefinition(
            role=role_name,
            meta_description=meta_description or current.meta_description,
            role_instructions=role_instructions,
        )
        replacement = tuple(
            role if item.role == role_name else item
            for item in self.roles
        )
        validate_role_definitions(replacement)
        self.store.stage_role_revision(
            role=role,
            active=True,
            operation="update",
            created_by_run_id=self.run_id,
            guidance=arguments,
        )
        self._working_roles[role.role] = role
        return AgentCreatorToolResult(
            status="ok",
        )

    def _prepare_author_job(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        call_index: int,
        reserved_role_names: set[str],
        reserved_role_count: int,
    ) -> "_AuthorJob":
        if tool_name == "add":
            role_name = _required_string(arguments, "role_name")
            guidance = _required_string(arguments, "instruction_guidance")
            meta_description = _required_string(arguments, "meta_description")
            if role_name in reserved_role_names:
                raise _ToolCallFailure("duplicate_role")
            if reserved_role_count >= self.max_roles:
                raise _ToolCallFailure("max_roles_reached")
            return _AuthorJob(
                tool_name=tool_name,
                arguments=arguments,
                call_index=call_index,
                author_input=RoleAuthorInput(
                    role_name=role_name,
                    instruction_guidance=guidance,
                    general_system_prompt=self.general_system_prompt,
                    metadata={"operation": "add", "run_id": self.run_id},
                ),
                meta_description=meta_description,
                reserve_role_name=role_name,
                reserve_role_count=1,
            )
        if tool_name == "update":
            role_name = _required_string(arguments, "role_name")
            failures = _required_string(arguments, "identified_failures")
            current = self._working_roles.get(role_name)
            if current is None:
                raise _ToolCallFailure("unknown_role")
            return _AuthorJob(
                tool_name=tool_name,
                arguments=arguments,
                call_index=call_index,
                author_input=RoleAuthorInput(
                    role_name=role_name,
                    identified_failures=failures,
                    current_role=current,
                    general_system_prompt=self.general_system_prompt,
                    metadata={"operation": "update", "run_id": self.run_id},
                ),
                meta_description=_optional_string(arguments, "meta_description"),
            )
        raise _ToolCallFailure("unknown_tool")

    def _run_author_jobs(self, jobs: list["_AuthorJob"]) -> list[str]:
        if not jobs:
            return []
        max_workers = max(1, min(len(jobs), self.max_tool_calls))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures: dict[Future[str], _AuthorJob] = {}
            for job in jobs:
                if job.tool_name == "add":
                    futures[
                        executor.submit(
                            self.role_author.create_role_instructions,
                            job.author_input,
                        )
                    ] = job
                elif job.tool_name == "update":
                    futures[
                        executor.submit(
                            self.role_author.update_role_instructions,
                            job.author_input,
                        )
                    ] = job
        results: list[str] = []
        for job in jobs:
            future = next(item for item, value in futures.items() if value is job)
            try:
                instructions = future.result()
                result = self._stage_authored_job(job, instructions)
            except Exception as exc:
                del exc
                result = AgentCreatorToolResult(status="failed", reason="tool_error")
            self._record(job.tool_name, job.arguments, job.call_index, result)
            results.append(_tool_result_text(result))
        return results

    def _stage_authored_job(
        self,
        job: "_AuthorJob",
        role_instructions: str,
    ) -> AgentCreatorToolResult:
        if job.tool_name == "add":
            role = AgentRoleDefinition(
                role=job.author_input.role_name,
                meta_description=job.meta_description,
                role_instructions=role_instructions,
            )
            validate_role_definitions((*self.roles, role))
            self.store.stage_role_revision(
                role=role,
                active=True,
                operation="add",
                created_by_run_id=self.run_id,
                guidance=job.arguments,
            )
            self._working_roles[role.role] = role
            return AgentCreatorToolResult(status="ok")
        if job.tool_name == "update":
            current = self._working_roles.get(job.author_input.role_name)
            if current is None:
                return AgentCreatorToolResult(status="failed", reason="unknown_role")
            role = AgentRoleDefinition(
                role=job.author_input.role_name,
                meta_description=job.meta_description or current.meta_description,
                role_instructions=role_instructions,
            )
            replacement = tuple(
                role if item.role == role.role else item
                for item in self.roles
            )
            validate_role_definitions(replacement)
            self.store.stage_role_revision(
                role=role,
                active=True,
                operation="update",
                created_by_run_id=self.run_id,
                guidance=job.arguments,
            )
            self._working_roles[role.role] = role
            return AgentCreatorToolResult(status="ok")
        return AgentCreatorToolResult(status="failed", reason="unknown_tool")

    def _record(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        call_index: int,
        result: AgentCreatorToolResult,
    ) -> None:
        self.store.record_tool_call(
            run_id=self.run_id,
            call_index=call_index,
            tool_name=tool_name,
            arguments=arguments,
            ok=result.ok,
            result=_tool_result_payload(result),
            error=None if result.ok else result.reason,
        )


class _ToolCallFailure(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class _AuthorJob:
    def __init__(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        call_index: int,
        author_input: RoleAuthorInput,
        meta_description: str,
        reserve_role_name: str = "",
        reserve_role_count: int = 0,
    ) -> None:
        self.tool_name = tool_name
        self.arguments = arguments
        self.call_index = call_index
        self.author_input = author_input
        self.meta_description = meta_description
        self.reserve_role_name = reserve_role_name
        self.reserve_role_count = reserve_role_count


def _required_string(arguments: dict[str, Any], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value.strip():
        raise _ToolCallFailure("invalid_arguments")
    return value.strip()


def _optional_string(arguments: dict[str, Any], key: str) -> str:
    value = arguments.get(key)
    if value is None:
        return ""
    if not isinstance(value, str):
        raise _ToolCallFailure("invalid_arguments")
    return value.strip()


def _mutation_tool_call(mutation: CreatorMutation) -> tuple[str, dict[str, Any]]:
    if mutation.action == "delete":
        return "delete", {"role_name": mutation.role_name}
    if mutation.action == "add":
        return (
            "add",
            {
                "role_name": mutation.role_name,
                "instruction_guidance": mutation.instruction_guidance,
                "meta_description": mutation.meta_description,
            },
        )
    if mutation.action == "update":
        arguments = {
            "role_name": mutation.role_name,
            "identified_failures": mutation.identified_failures,
        }
        if mutation.meta_description:
            arguments["meta_description"] = mutation.meta_description
        return "update", arguments
    return mutation.action, {"role_name": mutation.role_name}


def _tool_result_text(result: AgentCreatorToolResult) -> str:
    return json.dumps(
        _tool_result_payload(result),
        ensure_ascii=False,
        sort_keys=True,
    )


def _tool_result_payload(result: AgentCreatorToolResult) -> dict[str, str]:
    payload = {"status": result.status}
    if not result.ok:
        payload["reason"] = result.reason
    return payload
